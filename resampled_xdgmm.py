import re
from logging import warning
from typing import Dict, List, Union
from warnings import warn

import ghalton
import os
from scipy.special import ndtri
from tqdm import tqdm

# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
import numpy as np
from scipy import stats

from jax.config import config
config.update("jax_enable_x64", True)  # must be turned on for numerical stability
import jax
import jax.numpy as jnp
from jax.scipy import linalg as jlinalg
from jax.scipy.special import logsumexp as jlogsumexp
from jax import vmap, pmap
import psutil
from functools import partial, wraps

@jax.jit
def jax_xdgmm_EMstep(X, Xerr, mu, V, alpha, w=0., extra_pi=None):
    n_samples, n_features = X.shape
    w = jnp.eye(n_features) * w
    n_components = len(alpha)
    X = X[:, None, :]
    Xerr = Xerr[:, None, :, :]
    w_m = X - mu
    T = Xerr + V
    # ------------------------------------------------------------
    # compute inverse of each covariance matrix T
    Tshape = T.shape
    T = T.reshape([n_samples * n_components,
                   n_features, n_features])
    Tinv = vmap(jnp.linalg.inv)(T).reshape(Tshape)
    T = T.reshape(Tshape)

    # ------------------------------------------------------------
    # evaluate each mixture at each point
    logN = jlog_multivariate_gaussian(X, mu, T)

    # ------------------------------------------------------------
    # E-step:
    #  compute q_ij, b_ij, and B_ij
    # q = (N * alpha) / jnp.dot(N, alpha)[:, None]
    logp = logN + jnp.log(alpha)
    total = jlogsumexp(logp, axis=1, keepdims=True)
    if extra_pi is not None:
        total = jlogsumexp(jnp.concatenate([total, extra_pi], axis=1), axis=1, keepdims=True)

    logq = logp - total

    tmp = jnp.sum(Tinv * w_m[:, :, None, :], -1)
    b = mu + jnp.sum(V * tmp[:, :, None, :], -1)

    tmp = jnp.sum(Tinv[:, :, :, :, None]
                 * V[:, None, :, :], -2)
    B = V - jnp.sum(V[:, :, :, None]
                        * tmp[:, :, None, :, :], -2)

    # ------------------------------------------------------------
    # M-step:
    #  compute alpha, m, V
    # qj = q.sum(0)
    logqj = jlogsumexp(logq, axis=0)

    alpha = jnp.exp(logqj - jnp.log(n_samples))

    qj = jnp.exp(logqj)
    q = jnp.exp(logq)

    mu = jnp.sum(q[:, :, None] * b, 0) / qj[:, None]

    m_b = mu - b
    tmp = m_b[:, :, None, :] * m_b[:, :, :, None]
    tmp += B
    tmp *= q[:, :, None, None]
    V = tmp.sum(0) / qj[:, None, None]
    return mu, V + w, alpha, jlogsumexp(jnp.concatenate([logp, extra_pi], axis=1), axis=1).sum() + vmap(lambda v: lnprior(v, w))(V).sum()

def pmap_batched(function, nprocesses=None, *args, **kwargs):
    """
    Combines pmap looping to allow easy parallel use with batches over the first axes
    input nonstatic shapes = (nbatch, n
    """
    if nprocesses is None:
        possible = re.findall(r'--xla_force_host_platform_device_count=(\d+)', os.environ.get('XLA_FLAGS', ''))
        if not len(possible):
            print('xla_force_host_platform_device_count in XLA_FLAGS not set')
            nprocesses = psutil.cpu_count() - 1
        else:
            nprocesses = int(possible[-1])
        # warn(f'Using {nprocesses} processes without user instruction')
    p = pmap(function, *args, **kwargs)
    statics = kwargs.get('static_broadcasted_argnums', [])
    @wraps(p)
    def inner(*args):
        for i in range(len(args)):
            if i not in statics:
                break
        bsize = nprocesses
        if len(args[i]) <= bsize:
            return p(*args)
        nbatches, r = divmod(len(args[i]), bsize)
        nbatches += r > 0

        stack = []
        for b in range(nbatches):
            current_args = [a if i in statics else a[b*bsize:(b*bsize)+bsize] for i, a in enumerate(args)]
            stack.append(p(*current_args))
        if len(stack) == 1:
            return stack[0]
        try:
            return jnp.concatenate(stack, axis=0)
        except (TypeError, ValueError):
            return [jnp.concatenate(x, axis=0) for x in zip(*stack)]
    return inner


def sigmoid(x, scale=1000., floor=1e-5):
    ba = 1 - floor
    return ((0.5 * (jnp.tanh(x * scale / 2) + 1)) * ba) + floor


def above_lower(x, l, scale=1000., floor=1e-5):
    return sigmoid(x - l, scale, floor)


def below_upper(x, u, scale=1000., floor=1e-5):
    return 1 - sigmoid(x - u, scale, floor)


def combine_soft_filters(a):
    return jnp.prod(jnp.stack(a), axis=0)


def ndim_soft_filters(x, limits, scale=1000., floor=1e-5):
    """
    Use a sigmoid function instead of a boolean filter to define a binary yes/no that a point x
    is within the limits.
    limits.shape == (ndim, 2) where index 0 is the lower and index 1 is the upper limit
    this resolves the differentiation problem
    """
    aboves = [above_lower(x[..., i], l, scale, floor) for i, (l, u) in enumerate(limits)]
    belows = [below_upper(x[..., i], u, scale, floor) for i, (l, u) in enumerate(limits)]
    return combine_soft_filters(aboves+belows)


def jlog_multivariate_gaussian(x, mu, V):
    ndim = x.shape[-1]
    x_mu = x - mu

    if V.shape[-2:] != (ndim, ndim):
        raise ValueError("Shape of (x-mu) and V do not match")

    Vshape = V.shape
    V = V.reshape([-1, ndim, ndim])

    # Vchol = jnp.array([jlinalg.cholesky(V[i], lower=True)
    #                       for i in range(V.shape[0])])
    Vchol = vmap(partial(jlinalg.cholesky, lower=True))(V)

    # we may be more efficient by using scipy.linalg.solve_triangular
    # with each cholesky decomposition
    VcholI = vmap(jlinalg.inv)(Vchol)
    # VcholI = jnp.array([jlinalg.inv(_v) for _v in Vchol])
    logdet = vmap(lambda _v: 2 * jnp.sum(jnp.log(jnp.diagonal(_v))))(Vchol)
    # logdet = jnp.array([2 * jnp.sum(jnp.log(jnp.diagonal(Vchol[i])))
    #                        for i in range(V.shape[0])])

    VcholI = VcholI.reshape(Vshape)
    logdet = logdet.reshape(Vshape[:-2])

    VcIx = jnp.sum(VcholI * x_mu.reshape(x_mu.shape[:-1]
                                         + (1,) + x_mu.shape[-1:]), -1)
    xVIx = jnp.sum(VcIx ** 2, -1)
    return -0.5 * ndim * jnp.log(2 * jnp.pi) - 0.5 * (logdet + xVIx)


def rho2unbounded(rho):
    return jnp.log(rho + 1) - jnp.log(1 - rho)


def unbounded2rho(u):
    return jnp.where(u < jnp.inf, (jnp.exp(u) - 1) / (jnp.exp(u) + 1), 1.)


def std2unbounded(s):
    return jnp.log(s)


def unbounded2std(u):
    return jnp.exp(u)


def cov2corr(cov):
    std_ = jnp.sqrt(jnp.diag(cov))
    corr = cov / jnp.outer(std_, std_)
    return corr, std_


def corr2cov(corr, std):
    return corr * jnp.outer(std, std)


def tril(m):
    """Extracts the lower triangle indices"""
    corr, std = cov2corr(m)
    offdiag = corr[jnp.tril_indices(corr.shape[0], k=-1)]
    unbounded = rho2unbounded(offdiag)
    return jnp.concatenate([unbounded, std2unbounded(std)], axis=0)
    # return jnp.array([m[i, j] for i in range(m.shape[0]) for j in range(m.shape[0]) if i <= j])
    # return m[jnp.tril_indices(m.shape[0])]


def invert_tril(a, ndim):
    """reasembles a symmetric matrix from flattened lower triangle entries"""
    corr = jnp.ones((ndim, ndim))
    indices = jnp.tril_indices(ndim, k=-1)
    rhos = unbounded2rho(a[:-ndim])
    corr = jax.ops.index_update(corr, indices, rhos)
    corr = jax.ops.index_update(corr.T, indices, rhos).T
    std = unbounded2std(a[-ndim:])
    return corr2cov(corr, std)

def fraction_not_truncated(mu, v, limits, stdnorm_samples, scale, floor):
    L = jnp.linalg.cholesky(v)
    y = vmap(lambda x: jnp.dot(L, x))(stdnorm_samples) + mu
    filt = ndim_soft_filters(y, limits, scale, floor)
    r, err = jnp.mean(filt, axis=0), jnp.std(filt, axis=0) / jnp.sqrt(stdnorm_samples.shape[0])
    return jnp.where(r < 0., 0., jnp.where(r > 1., 1., r)), err


def unnormed_two_gaussians_product(mu, v, datum, datumcov):
    invv = jnp.linalg.inv(v)
    invdatumcov = jnp.linalg.inv(datumcov)
    invc = invv + invdatumcov
    vc = jnp.linalg.inv(invc)
    mc = vmap(jnp.dot)(vc, jnp.dot(invv, mu) + vmap(jnp.dot)(invdatumcov, datum))
    return mc, vc


def truncation_norm(mu, v, datum, datumcov, limits, stdnorm_samples, scale, floor):
    mean, cov = unnormed_two_gaussians_product(mu, v, datum, datumcov)
    return fraction_not_truncated(mean, cov, limits, stdnorm_samples, scale, floor)


def truncated_pi(mu, v, data, datacov, stdnorm_samples, limits=None, dataweights=None):
    if dataweights is None:
        dataweights = 0.
    if limits is not None:
        # if norm ->-inf when model_norm -> 0, post_norm -> 0
        model_norm, model_norm_err = fraction_not_truncated(mu, v, limits, stdnorm_samples, 10000, 1e-12)
        post_norm, post_norm_err = truncation_norm(mu, v, data, datacov, limits, stdnorm_samples, 10000, 1e-12)
        # log_post_norm = jnp.where(post_norm / post_norm_err > 3, jnp.log(post_norm), -jnp.log(stdnorm_samples.shape[0]))
        # log_model_norm = jnp.where(model_norm / model_norm_err > 3, jnp.log(model_norm), -jnp.log(stdnorm_samples.shape[0]))
        # log_norm = log_post_norm - log_model_norm
        log_norm = jnp.log(post_norm) - jnp.log(model_norm)
    else:
        log_norm = 0.
    T = v[None, ...] + datacov
    return (jlog_multivariate_gaussian(data, mu, T) + log_norm) + dataweights


def truncated_pki(mus, Vs, alphas, data, datacov, stdnorm_samples, limits=None, dataweights=None, nprocesses=None):
    """Return P(x|gaussian) for each gaussian and each x, returned shape = (ngaussians, nresamples, nx)"""
    if limits is not None:
        static = (2, 3, 4, 6)
    else:
        static = (2, 3, 4, 5, 6)
    return pmap_batched(truncated_pi, nprocesses, static_broadcasted_argnums=static)(mus, Vs, data, datacov, stdnorm_samples, limits, dataweights) \
                    + jnp.log(alphas[:, None])


def truncated_qki(mus, Vs, alphas, data, datacov, stdnorm_samples, limits=None, extra_pki=None, dataweights=None, return_all=False, nprocesses=None):
    pki = truncated_pki(mus, Vs, alphas, data, datacov, stdnorm_samples, limits, dataweights, nprocesses)
    if extra_pki is None:
        _pki = pki
    else:
        _pki = jnp.concatenate([pki]+extra_pki, axis=0)
    total = jlogsumexp(_pki, axis=0, keepdims=True)
    if dataweights is None:
        dw = 1.
    else:
        dw = jnp.exp(dataweights)
    qki = jnp.where(total > -jnp.inf, pki - total, -jnp.inf) * dw
    if return_all:
        return qki, pki, total
    return qki


def lnprior(v, w):
    return -jnp.trace(jnp.linalg.inv(v) * w)


def negative_complete_loglike(mu, v, qi, data, datacov, stdnorm_samples, limits=None, dataweights=None):
    """
    -lnL,
    lower/more negative is a better fit
    """
    pi = truncated_pi(mu, v, data, datacov, stdnorm_samples, limits, dataweights)  # P(x_i|theta)
    # because if we do this for all gaussians the logsumexp takes care of the -infs (each data point must have at least one gaussian)
    # so in the per gaussian likelihood, we can ignore the points which don't belong (-infs)
    return -jnp.sum(jnp.where(jnp.isfinite(qi), pi * jnp.exp(qi), 0.))


def objective(p, ndim, qi, data, datacov, w, stdnorm_samples, limits=None, dataweights=None):
    mu = p[:ndim]
    v = invert_tril(p[ndim:], ndim)
    return negative_complete_loglike(mu, v, qi, data, datacov, stdnorm_samples, limits, dataweights) - lnprior(v, w)

def prior(p, ndim, qi, data, datacov, w, stdnorm_samples, limits=None, dataweights=None):
    mu = p[:ndim]
    v = invert_tril(p[ndim:], ndim)
    return jlog_multivariate_gaussian(mu, 0, np.eye(ndim)*10) - lnprior(v, w)


def regularised_update(ndim, initial, gradient, learning_rate, max_mean_jumps, max_std_scales):
    """
    Puts a hard bound on how far you can jump in one movement
    transforms:
        mu: no transform
        std: log(std)
    max jump in mu = additive
    max jump in v = multiplicative
    """
    proposed_jump = gradient * learning_rate
    proposed_update = initial - gradient * learning_rate
    # return proposed_update, False

    mean_check = proposed_jump[:ndim] > max_mean_jumps
    std_check = (unbounded2std(proposed_update[-ndim:]) / unbounded2std(initial[-ndim:])) > max_std_scales

    moderated_mean_update = jnp.where(mean_check,
                                      max_mean_jumps+initial[:ndim],
                                      proposed_update[:ndim])
    moderated_std_update = jnp.where(std_check,
                                     std2unbounded(unbounded2std(initial[-ndim:]) * max_std_scales),
                                     proposed_update[-ndim:])
    proposed_update = jax.ops.index_update(proposed_update, np.array(range(ndim)), moderated_mean_update)
    proposed_update = jax.ops.index_update(proposed_update, np.array(range(-ndim, 0)), moderated_std_update)
    return proposed_update, jnp.any(mean_check) | jnp.any(std_check)


def mstep_1gaussian(ndim, qi, current_mu, current_V, data, datacov, w, stdnorm_samples,
                    learning_rate, limits=None, dataweights=None):
    _v = tril(current_V)
    p0 = jnp.concatenate([current_mu, _v], axis=0)
    grad = jax.grad(objective)(p0, ndim, qi, data, datacov, w, stdnorm_samples, limits, dataweights)
    prior_grad = jax.grad(prior)(p0, ndim, qi, data, datacov, w, stdnorm_samples, limits, dataweights)
    bad_grad = jnp.isnan(grad)
    grad = jnp.where(bad_grad, prior_grad, grad)
    update, regularised = regularised_update(ndim, p0, grad, learning_rate, data.std(axis=0)*4, 3.)
    mu, v = update[:ndim], invert_tril(update[ndim:], ndim)
    return mu, v, jnp.any(bad_grad), regularised


def mstep_alphas(qki, dirichlet_prior=1.):
    """alpha_k = sum_i(qki) / N"""
    qki = jnp.logaddexp(qki, jnp.log(dirichlet_prior-1))
    a = jlogsumexp(qki, axis=-1) - jnp.log(qki.shape[1]) - jnp.log(dirichlet_prior * qki.shape[0]) - jnp.log(qki.shape[0])
    return jnp.exp(jnp.where(a > -jnp.inf, a, -23.))  # set really small


def mstep_muv(ndim, qki, current_mus, current_Vs, data, datacov, w, stdnorm_samples, learning_rate,
              limits=None, dataweights=None, nprocesses=None):
    if limits is not None:
        static = (0, 4, 5, 6, 7, 8, 10)
    else:
        static = (0, 4, 5, 6, 7, 8, 9, 10)
    mus, Vs, bad_grads, regularised = pmap_batched(mstep_1gaussian, nprocesses, static_broadcasted_argnums=static)(ndim, qki, current_mus, current_Vs,
                                                                                data, datacov, w,
                                                                       stdnorm_samples, learning_rate, limits,
                                                                       dataweights)
    # f = vmap(lambda qi, mu, v, lim: mstep_1gaussian(ndim, qi, mu, v, data, datacov, w, stdnorm_samples, learning_rate, lim, dataweights))
    # mus, Vs, bad_grads, regularised = f(qki, current_mus, current_Vs, limits)
    return mus, Vs, jnp.sum(bad_grads), jnp.sum(regularised)


def scale_groups(current_alphas, group_ratios):
    """
    Ensure that the ratios of each of the components in a given group remain the same
    For each group:
        sum total weight after gradient descent
        multiply by given ratios
    """
    for group in group_ratios:
        total_weight = jnp.sum(current_alphas[group > 0])
        current_alphas = jnp.where(group > 0, total_weight * group, current_alphas)
    return current_alphas

def postfix_components(current_mus, current_Vs, current_alphas, fix_components, initial_mus, initial_Vs, initial_alphas, extra_weight=0.):
    total_fixed_weight = 0
    free_alphas = []
    for k in range(initial_mus.shape[0]):
        params = fix_components.get(k, [])
        if 'mu' in params or 'all' in params:
            current_mus = jax.ops.index_update(current_mus, k, initial_mus[k])
        if 'v' in params or 'all' in params:
            current_Vs = jax.ops.index_update(current_Vs, k, initial_Vs[k])
        if 'weight' in params or 'all' in params or 'weights' in params or 'alpha' in params:
            current_alphas = jax.ops.index_update(current_alphas, k, initial_alphas[k])
            total_fixed_weight = total_fixed_weight + current_alphas[k]
        else:
            free_alphas.append(k)
    total_free_weight = current_alphas[np.array(free_alphas)].sum()
    adjusted_free_weight = 1 - total_fixed_weight
    current_alphas = jax.ops.index_mul(current_alphas, np.array(free_alphas), adjusted_free_weight / total_free_weight)
    current_alphas = current_alphas / current_alphas.sum() * (1 - extra_weight)
    return current_mus, current_Vs, current_alphas


def one_step(ndim, mu, v, weights, data, datacov, w,  stdnorm_samples, learning_rate, limits, extra_pki, dataweights, nprocesses=None):
    qki, pki, normi = truncated_qki(mu, v, weights, data, datacov, stdnorm_samples, limits,
                                    extra_pki, dataweights, return_all=True, nprocesses=nprocesses)
    _mu, _v, bad_grads, regularised = mstep_muv(ndim, qki, mu, v, data, datacov, w, stdnorm_samples,
                                                learning_rate, limits, dataweights, nprocesses)
    return _mu, _v, qki, pki, normi, jnp.sum(bad_grads), jnp.sum(regularised)

def run(n: int, mu: np.ndarray, v: np.ndarray, weights: np.ndarray, data: np.ndarray,
        datacov: np.ndarray, w: float, stdnorm_samples: np.ndarray, dataweights: np.ndarray = None,
        learning_rate=1e-5,
        limits: np.ndarray = None,
        fix_components: Dict[int, Union[List[str], str]] = None, groups: List[List[int]] = None,
        extra_component_pi: np.ndarray = None, extra_weight: float = 0.,
        tol: float = 1e-6, patience: int = 10, nprocesses=None,
        validation_data=None, validation_datacov=None, validation_dataweights=None,
        dirichlet_prior=1.,
        printit=False, no_prints=False, previous_data=None):
    if fix_components is None:
        fix_components = {}
    else:
        fix_components = {int(k): [v] if isinstance(v, str) else v for k, v in fix_components.items()}
        for vs in fix_components.values():
            assert all(v in ['mu', 'v', 'weight', 'all'] for v in vs), "fixed parameters must be mu,v,weight or 'all'"
    if groups is None:
        groups = []
    mu =  np.asarray(mu).astype(float)
    v =  np.asarray(v).astype(float)
    weights =  np.asarray(weights).astype(float)
    data =  np.asarray(data).astype(float)
    datacov =  np.asarray(datacov).astype(float)
    stdnorm_samples =  np.asarray(stdnorm_samples).astype(float)
    limits = np.asarray(limits).astype(float)
    if dataweights is not None:
        dataweights = np.asarray(dataweights).astype(float)
    if extra_component_pi is not None:
        extra_component_pi = np.asarray(extra_component_pi).astype(float)
    extra_weight = float(extra_weight)
    w = float(w)
    n = int(n)
    ndim = mu.shape[1]
    ncomp = weights.shape[0]
    ndata = data.shape[0]
    assert mu.shape == (ncomp, ndim), f"mu.shape != (ncomp, ndim) {mu.shape} != {(ncomp, ndim)}"
    assert v.shape == (ncomp, ndim, ndim), f"v.shape != (ncomp, ndim, ndim) {v.shape} != {(ncomp, ndim, ndim)}"
    assert weights.shape == (ncomp,), f"weights.shape != (ncomp,) {weights.shape} != {(ncomp,)}"
    assert data.shape == (ndata, ndim), f"data.shape != (ndata, ndim) {data.shape} != {(ndata, ndim)}"
    assert datacov.shape == (ndata, ndim, ndim), f"datacov.shape != (ndata, ndim, ndim) {datacov.shape} != {(ndata, ndim, ndim)}"
    assert stdnorm_samples.shape[1:] == (ndim, ), f"stdnorm_samples.shape[1:] != (ndim, ) {stdnorm_samples.shape[1:]} != {(ndim, )}"
    assert limits.shape == (ncomp, ndim, 2), f"limits.shape != (ncomp, ndim, 2) {limits.shape} != {(ncomp, ndim, 2)}"
    if dataweights is not None:
        assert dataweights.shape == (ndata, ), f"dataweights.shape != (ndata, ) {dataweights.shape} != {(ndata,)}"
    if extra_component_pi is not None:
        assert extra_component_pi.shape == (ndata, ), f"disjoint_component_pi.shape != (ndata, ) {extra_component_pi.shape} != {(ndata,)}"
    if extra_component_pi is None:
        assert np.allclose(weights.sum(), 1)
    extra_pki = []
    if extra_component_pi is not None:
        extra_pki.append(extra_component_pi[None, :])
    free_components = ~np.array([all(n in fix_components.get(i, []) for n in ['mu', 'v', 'weight']) or 'all' in fix_components.get(i, [])
                          for i in range(ncomp)])
    free_components_index = np.where(free_components)[0]
    fully_fixed_index = np.where(~free_components)[0]
    free_muv = ~np.array([all(n in fix_components.get(i, []) for n in ['mu', 'v']) or 'all' in fix_components.get(i, [])
                                for i in free_components_index])
    free_muv_index = np.where(free_muv)[0]
    free_weights = ~np.array([any(v in fix_components.get(i, []) for v in ['weight', 'all'])
                                for i in free_components_index])
    free_weights_index = np.where(free_weights)[0]
    if not no_prints:
        if len(fix_components):
            for i in range(ncomp):
                print(f'fixing {fix_components.get(i, [])} of component {i}')
        if len(fully_fixed_index):
            print(f'Only initial logp computation will happen for components {fully_fixed_index.tolist()} ({weights[~free_components].sum():.2f} weight fixed)')
        if len(free_muv_index):
            print(f'Gradient descent of mu & V will be performed for components {free_components_index[free_muv_index].tolist()}')
        if len(free_weights_index):
            print(f'Analytical minimisation of component weight will be performed for components {free_components_index[free_weights_index].tolist()}')
    if sum(~free_components):
        # take out fully fixed components (mu, v, alpha) and give only the pki and extra_weight
        for i, group in list(enumerate(groups)):
            fixed_check = [j for j, g in enumerate(group) if j in fully_fixed_index.tolist()]
            if len(fixed_check) == len(group):
                del groups[i]
            elif len(fixed_check):
                raise ValueError(f"Cannot fully fix components {fixed_check} when they are included in a group")
        fixed_pki = truncated_pki(mu[~free_components], v[~free_components], weights[~free_components],
                                  data, datacov, stdnorm_samples, limits[~free_components], dataweights, nprocesses)
        extra_pki.append(fixed_pki)
        extra_weight += weights[~free_components].sum()
    for p in extra_pki:
        assert np.all(np.isfinite(p)), "Not all extra pki were finite"
    group_ratios = np.zeros((len(groups), ncomp))
    for i, group in enumerate(groups):
        for comp in group:
            group_ratios[i, comp] = weights[comp]
    group_ratios /= group_ratios.sum(axis=1, keepdims=True)
    if group_ratios.size:
        assert np.all(np.allclose(group_ratios.sum(axis=1), 1)), "A group must have weights sum to 1"
        assert np.all(np.sum(group_ratios > 0, axis=0) <= 1), "A component must belong to exactly one group or no group"
    if groups is not None:
        for group, ratio in zip(groups, group_ratios):
            ratio = ':'.join([f"{ratio[k]/ratio[group[0]]:.2f}" for k in group])
            print(f'components {group} will have their weight ratio ({ratio}) fixed wrt each other')
    np.testing.assert_equal(np.isfinite(mu), True, err_msg='not all input mu are finite')
    np.testing.assert_equal(np.isfinite(v), True, err_msg='not all input v are finite')
    np.testing.assert_equal(np.isfinite(weights), True, err_msg='not all input weights are finite')
    np.testing.assert_equal(np.isfinite(data), True, err_msg='not all input data are finite')
    np.testing.assert_equal(np.isfinite(datacov), True, err_msg='not all input datacov are finite')
    np.testing.assert_equal(np.isfinite(stdnorm_samples), True, err_msg='not all input stdnorm_samples are finite')
    fix_components = {k: v for k, v in fix_components.items() if k in free_components_index}
    mus, vs, alphas, logls, orphans = _run(n,
                mu[free_components_index], v[free_components_index], weights[free_components_index],
                data, datacov, w, stdnorm_samples, dataweights, learning_rate,
                limits[free_components_index], fix_components, free_muv_index, free_weights_index,
                group_ratios[:, free_components_index], extra_pki, extra_weight, tol, patience,
                nprocesses, validation_data, validation_datacov, validation_dataweights, dirichlet_prior,
                                           printit, no_prints, previous_data)
    for i in fully_fixed_index:
        mus = np.append(mus, np.ones((mus.shape[0], 1, ndim)) * mu[i], axis=1)
        vs = np.append(vs, np.ones((vs.shape[0], 1, ndim, ndim)) * v[i], axis=1)
        alphas = np.append(alphas, np.ones((alphas.shape[0], 1)) * weights[i], axis=1)
    resort = np.argsort(free_components_index.tolist() + fully_fixed_index.tolist())
    return mus[:, resort], vs[:, resort], alphas[:, resort], logls, orphans


def _run(n: int, mu: np.ndarray, v: np.ndarray, weights: np.ndarray, data: np.ndarray,
         datacov: np.ndarray, w: float, stdnorm_samples: np.ndarray, dataweights: Union[np.ndarray, None],
         learning_rate: float,
         limits: np.ndarray,
         fix_components: Dict[int, Union[List[str], str]],
         free_muv_index: np.ndarray, free_weights_index: np.ndarray,
         group_ratios: np.ndarray,
         extra_pki: List[np.ndarray], extra_weight: float,
         tol: float, patience: int, nprocesses: int = None,
         validation_data=None, validation_datacov=None, validation_dataweights=None,
         dirichlet_prior=1.,
         printit=False, no_prints=False, previous_data=None):
    initial_mu = mu.copy()
    initial_v = v.copy()
    initial_weights = weights.copy()
    ndim = mu.shape[1]
    ncomp = weights.shape[0]

    mus = np.zeros((n, ncomp, ndim))
    vs = np.zeros((n, ncomp, ndim, ndim))
    alphas = np.zeros((n, ncomp))
    orphans = np.zeros((n,))
    logls = jnp.zeros(n)

    if previous_data is not None:
        mus = np.concatenate([previous_data[0], mus], axis=0)
        vs = np.concatenate([previous_data[1], vs], axis=0)
        alphas = np.concatenate([previous_data[2], alphas], axis=0)
        orphans = np.concatenate([previous_data[3], orphans], axis=0)
        logls = np.concatenate([previous_data[4], logls], axis=0)


    bar = tqdm(range(n))
    for i in bar:
        if previous_data is not None:
            i += len(previous_data[0])
        qki, pki, normi = truncated_qki(mu, v, weights, data, datacov, stdnorm_samples, limits,
                                        extra_pki, dataweights, return_all=True, nprocesses=nprocesses)
        _mu, _v, bad_grads, regularised = mstep_muv(ndim, qki[free_muv_index],
                                                    mu[free_muv_index], v[free_muv_index],
                                                    data, datacov, w, stdnorm_samples,
                                                    learning_rate, limits[free_muv_index],
                                                    dataweights, nprocesses)
        _weights = mstep_alphas(qki[free_weights_index], dirichlet_prior)
        if np.any(~np.isfinite(_v)) or np.any(~np.isfinite(_mu)):
            bad_comps = np.where((np.any(~np.isfinite(_v), axis=(1, 2)) |
                                  np.any(~np.isfinite(_mu), axis=1)))[0]
            bad_qki = np.where(~np.isfinite(qki))
            # bad_qki_str = '\n'.join([f"k={a};i={b} q={qki[a, b]}" for a, b in zip(*bad_qki)])
            raise ValueError(f"Non finite value reached for comps={free_muv_index[bad_comps]}:\ninput:\n"
                             f"mu={mu[free_muv_index][bad_comps]}\n"
                             f"v={v[free_muv_index][bad_comps]}\n"
                             f"w={weights[free_muv_index][bad_comps]}\n"
                             f"output\n"
                             f"mu={_mu[bad_comps]}\nv={_v[bad_comps]}\nw={_weights[bad_comps]}\n"
                             f"bad qki {len(bad_qki[0])}")
        elif np.any(~np.isfinite(_weights)):
            bad_comps = np.where(~np.isfinite(_weights))[0]
            bad_qki = np.where(~np.isfinite(qki))
            # bad_qki_str = '\n'.join([f"k={a};i={b} q={qki[a, b]}" for a, b in zip(*bad_qki)])
            raise ValueError(f"Non finite value reached for comps={free_weights_index[bad_comps]}:\ninput:\n"
                             f"w={weights[free_weights_index][bad_comps]}\n"
                             f"output\n"
                             f"w={_weights[bad_comps]}\n"
                             f"bad qki {len(bad_qki[0])}")

        # put changed parameters back
        mu = jax.ops.index_update(mu, free_muv_index, _mu)
        v = jax.ops.index_update(v, free_muv_index, _v)
        weights = jax.ops.index_update(weights, free_weights_index, _weights)
        # scale intra-group weights
        weights = scale_groups(weights, group_ratios)
        # fix back unvectored parameters (i.e. a single mu but not V)
        mu, v, weights = postfix_components(mu, v, weights, fix_components, initial_mu, initial_v, initial_weights, extra_weight)
        assert np.allclose(weights.sum()+extra_weight, 1.)

        # summary stuff
        check = normi > -jnp.inf
        if validation_data is None:
            logl = jnp.sum(jnp.where(check, normi, 0.)) + jnp.sum(vmap(partial(lnprior, w=w))(v))
        else:
            logl = jlogsumexp(truncated_pki(mu, v, weights, validation_data, validation_datacov,
                                            stdnorm_samples, limits, validation_dataweights, nprocesses), axis=0).sum()
            logl = logl + jnp.sum(vmap(partial(lnprior, w=w))(v))
        norphans = jnp.sum(~check)
        if printit:
            with np.printoptions(precision=3, suppress=True):
                for k in range(ncomp):
                    print(f"alpha{k}={weights[k]:.3f}")
                    print(f"mu{k}={mu[k]}")
                    print(f"v{k}={v[k]}")
                print(f"logl={logl:.3f}")
                print(f"orphaned data: {norphans:.0f}")
                print(f"bad grads: {bad_grads:.0f}")
                print(f"regularised: {regularised:.0f}")
                print(f'====')
        logls = jax.ops.index_update(logls, i, logl)
        mus = jax.ops.index_update(mus, i, mu)
        vs = jax.ops.index_update(vs, i, v)
        alphas = jax.ops.index_update(alphas, i, weights)
        orphans = jax.ops.index_update(orphans, i, norphans)
        last_index = 0
        if i > 0:
            if not last_index % patience:
                last_index = i
            diff = float((logls[i] - logls[i-1]) / np.abs(logls[i-1]))
            diff_bar = float((logls[i] - logls[last_index]) / np.abs(logls[last_index]))
            diff_patience = float((logls[i] - logls[i - patience]) / np.abs(logls[i - patience]))
            if not i % patience:
                if diff_patience < tol:
                    print(f"Terminated early due to lack of patience over {patience} iterations")
                    besti = np.argmax(logls[:i])
                    print(f"Erasing last {i-besti} iteration since they didn't yield anything")
                    mus, vs, alphas, logls, orphans = mus[:besti], vs[:besti], alphas[:besti], logls[:besti], orphans[:besti]
                    break
            sign = np.sign(diff_bar - tol)
            sign = '+' if sign else '-' if sign < 0 else '?'
            bar.desc = f"{sign}|dlogl={diff:+.2g}|pr'd={bad_grads}|reg'd={regularised}"
    if previous_data is not None:
        l = len(previous_data[0])
    else:
        l = 0
    return np.asarray(mus)[l:], np.asarray(vs)[l:], np.asarray(alphas)[l:], np.asarray(logls)[l:], np.asarray(orphans)[l:]


def resample_data(x, err, n):
    return stats.norm(x, err).rvs((n,)+x.shape)


def generate_std_norm_samples(n, ndim, seed=None):
    gen = ghalton.Halton(int(ndim))
    if seed:
        gen.seed(seed)
    return ndtri(np.asarray(gen.get(int(n))))


# TODO: add optimiser usage (partiioned and non-partioned)
# TODO: shard static variables (data, datacov, datalimits, etc) to MPIJax and exec functions on batches of data rather than components (remove pmap)