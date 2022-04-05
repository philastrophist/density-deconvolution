from jax.config import config
config.update("jax_enable_x64", True)  # must be turned on for numerical stability

import h5py
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import jax.scipy.special
from astroML.density_estimation import XDGMM as astroML_XDGMM
import numpy as np
import warnings
from jax.scipy.special import logsumexp as jlogsumexp
import jax.numpy as jnp
from scipy import stats
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture as skl_GMM
from tqdm import tqdm
from xdgmm import XDGMM
from extreme_deconvolution import extreme_deconvolution\
                as bovyXD

from resampled_xdgmm import generate_std_norm_samples, run, truncated_qki, truncated_pki, jlog_multivariate_gaussian


def sample_truncated_gaussian(mu, cov, limits, n, random_state=None):
    _n = n
    mvn = stats.multivariate_normal(mu, cov)
    ndim = mu.shape[0]
    samples = np.zeros((0, ndim), dtype=float)
    while len(samples) < n:
        _samples = mvn.rvs(int(_n), random_state).reshape(-1, ndim)
        filt = np.all((_samples > limits[:, 0]) & (_samples < limits[:, 1]), axis=1)
        frac = filt.sum() / _n
        if frac == 0:
            _n = _n * 100
        else:
            _n = (n - len(samples)) / frac
        samples = np.concatenate([samples, _samples[filt]], axis=0)
    return samples[:n]


class GeneralisedXDGMM(XDGMM):
    def __init__(self, n_components=1, ndim=1, n_iter=0, tol=1E-5, learning_rate=1e-4,
                 patience=10, labels=None, random_state=None,
                 V=None, mu=None, weights=None, extra_weight: float = 0.,
                 limits=None, n_stdnorm_samples=9000, fix_components: Dict[int, List[str]] = None,
                 groups: List[List[int]] = None,
                 filename=None, w=0., dirichlet_prior=1., nprocesses=None):
        self._mu, self._V, self._weights = None, None, None
        self.GMM = astroML_XDGMM(n_components,
                                 max_iter=n_iter, tol=tol,
                                 random_state=random_state)
        super().__init__(n_components, n_iter, tol, 'Bovy', labels, random_state, V, mu, weights, None, w)
        self.dirichlet_prior = dirichlet_prior
        self.filename = None
        self.ndim = ndim
        self.patience = patience
        self.learning_rate = learning_rate
        self.limits = np.asarray(limits) if limits is not None else None
        self.extra_weight = extra_weight
        self.nprocesses = nprocesses
        self.n_stdnorm_samples = n_stdnorm_samples
        self.stdnorm_samples = generate_std_norm_samples(n_stdnorm_samples, self.ndim, self.random_state)
        self.fix_components = fix_components
        self.groups = groups
        self.orphans = 0
        self.clear_records()
        self.filename = filename
        if self.filename is not None:
            if Path(self.filename).exists():
                print(f'Retrieving previous parameters from {self.filename}')
                self.read_model(self.filename)
        initials = {}
        for n, x in zip(['mus', 'Vs', 'weightss'], [mu, V, weights]):
            if x is not None:
                initials[n] = np.asarray(x)
        self._append_records(**initials)

    @staticmethod
    def _read_init_args(filename):
        with h5py.File(filename, 'r') as f:
            n_components = f.attrs['n_components']
            ndim = f.attrs['ndim']
            n_iter = f.attrs['n_iter']
            tol = f.attrs['tol']
            learning_rate = f.attrs['learning_rate']
            patience = f.attrs['patience']
            labels = f.attrs.get('labels', None)
            random_state = f.attrs.get('random_state', None)
            extra_weight = f.attrs['extra_weight']
            limits = np.asarray([i.tolist() for i in f.attrs.get('limits', [])])
            if not len(limits):
                limits = None
            n_stdnorm_samples = f.attrs['n_stdnorm_samples']
            if 'fix_components_keys' in f.attrs.keys():
                fix_components = {int(k): v.split(' ') for k, v in zip(f.attrs['fix_components_keys'], f.attrs['fix_components_vals'])}
            else:
                fix_components = None
            groups = f.attrs.get('groups', None)
            w = f.attrs['w']
            dirichlet_prior = f.attrs.get('dirichlet_prior', 1.)
        return n_components, ndim, n_iter, tol, learning_rate, patience, labels, random_state, extra_weight,\
                limits, n_stdnorm_samples, fix_components, groups, w, dirichlet_prior

    def _write_init_args(self, filename):
        with h5py.File(filename, 'a') as f:
            f.attrs['n_components'] = self.n_components
            f.attrs['ndim'] = self.ndim
            f.attrs['n_iter'] = self.n_iter
            f.attrs['tol'] = self.tol
            f.attrs['learning_rate'] = self.learning_rate
            f.attrs['patience'] = self.patience
            if self.labels:
                f.attrs['labels'] = self.labels
            if self.random_state:
                f.attrs['random_state'] = self.random_state
            f.attrs['extra_weight'] = self.extra_weight
            if self.limits is not None:
                f.attrs['limits'] = self.limits
            f.attrs['n_stdnorm_samples'] = self.n_stdnorm_samples
            if self.fix_components is not None:
                if len(self.fix_components):
                    f.attrs['fix_components_keys'], f.attrs['fix_components_vals'] = zip(*{int(k): ' '.join(v) for k, v in self.fix_components.items()}.items())
            if self.groups is not None:
                f.attrs['groups'] = self.groups
            f.attrs['w'] = self.w
            f.attrs['dirichlet_prior'] = self.dirichlet_prior

    def _match_init_args(self, filename):
        try:
            args = self._read_init_args(filename)
        except KeyError:
            return
        limits = args[9]
        if limits is not None:
            limits = limits.tolist()
        args = (args[0], args[1], args[6], args[8], limits, args[13], args[14])
        if isinstance(self.limits, np.ndarray):
            limits = self.limits.tolist()
        else:
            limits = self.limits
        if args != (self.n_components, self.ndim,  self.labels, self.extra_weight, limits, self.w, self.dirichlet_prior):
            raise IOError(f"{filename} is not compatible with {self}")

    @staticmethod
    def _read_state(filename):
        with h5py.File(filename, 'r') as f:
            mu = f['mu'][:]
            V = f['V'][:]
            weights = f['weights'][:]
        return mu, V, weights

    def _write_state(self, filename):
        with h5py.File(filename, 'a') as f:
            f.require_dataset('mu', (self.n_components, self.ndim), dtype=float, data=self.mu)
            f['mu'][:] = self.mu
            f.require_dataset('V', (self.n_components, self.ndim, self.ndim), dtype=float, data=self.V)
            f['V'][:] = self.V
            f.require_dataset('weights', (self.n_components,), dtype=float, data=self.weights)
            f['weights'][:] = self.weights

    @staticmethod
    def _read_records(filename):
        with h5py.File(filename, 'r') as f:
            mus = f['mus'][:]
            Vs = f['Vs'][:]
            weightss = f['weightss'][:]
            orphanss = f['orphanss'][:].reshape(-1)
            logls = f['logls'][:].reshape(-1)
        return mus, Vs, weightss, logls, orphanss

    def _write_records(self, filename):
        with h5py.File(filename, 'a') as f:
            if 'mus' not in f.keys():
                f.create_dataset('mus', data=self.mus, maxshape=(None, self.n_components, self.ndim))
                f.create_dataset('Vs', data=self.Vs, maxshape=(None, self.n_components, self.ndim, self.ndim))
                f.create_dataset('weightss', data=self.weightss, maxshape=(None, self.n_components))
                try:
                    f.create_dataset('orphanss', data=self.orphanss, maxshape=(None, 1))
                except ValueError:
                    f.create_dataset('orphanss', data=self.orphanss, maxshape=(None, ))
                try:
                    f.create_dataset('logls', data=self.logls, maxshape=(None, 1))
                except ValueError:
                    f.create_dataset('logls', data=self.logls, maxshape=(None, ))
            else:
                for k in 'mus Vs weights orphanss logls'.split(' '):
                    l = f[k].shape[0]
                    data =  getattr(self, k)
                    assert len(data) >= l
                    if len(data) > l:
                        f[k].resize(data.shape[0], axis=0)
                        f[k][l:] = data[l:]

    def update_model(self, filename=None):
        filename = self.filename if filename is None else filename
        if Path(filename).exists():
            self._match_init_args(filename)
        self._write_init_args(filename)  # updates tol, niter
        self._write_records(filename)
        self._write_state(filename)

    def save_model(self, filename=None, overwrite=False):
        filename = self.filename if filename is None else filename
        if Path(filename).exists() and not overwrite:
            raise FileExistsError(f"{filename} already exists and overwrite=False")
        self._write_init_args(filename)
        self._write_records(filename)
        self._write_state(filename)

    def read_model(self, filename=None):
        filename = self.filename if filename is None else filename
        self._match_init_args(filename)
        self.mu, self.V, self.weights = self._read_state(filename)
        self.mus, self.Vs, self.weightss, self.logls, self.orphanss = self._read_records(filename)

    @classmethod
    def from_file(cls, filename, nprocesses=None):
        n_components, ndim, n_iter, tol, learning_rate, patience, labels, random_state, extra_weight, \
        limits, n_stdnorm_samples, fix_components, groups, w, dirichlet_prior = cls._read_init_args(filename)
        mu, V, weights = cls._read_state(filename)
        xdgmm = cls(n_components, ndim, n_iter, tol, learning_rate, patience, labels, random_state,
                    V, mu, weights, extra_weight, limits, n_stdnorm_samples, fix_components, groups,
                    filename, w, dirichlet_prior, nprocesses)
        xdgmm.clear_records()
        xdgmm._append_records(*cls._read_records(filename))
        xdgmm.mu, xdgmm.V, xdgmm.weights = cls._read_state(filename)
        return xdgmm

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value
        self.GMM.mu = value

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, value):
        self._V = value
        self.GMM.V = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value
        self.GMM.alpha = value

    @property
    def alpha(self):
        return self._weights

    @alpha.setter
    def alpha(self, value):
        self._weights = value
        self.GMM.alpha = value

    def clear_records(self):
        self.mus = np.zeros([0, self.n_components, self.ndim], dtype=float)
        self.Vs = np.zeros([0, self.n_components, self.ndim, self.ndim], dtype=float)
        self.weightss = np.zeros([0, self.n_components], dtype=float)
        self.logls = np.zeros([0, 1], dtype=float)
        self.orphanss = np.zeros([0, 1], dtype=float)

    def _append_records(self, mus=None, Vs=None, weightss=None, logls=None, orphanss=None):
        if mus is not None:
            if len(mus.shape) == 3:
                self.mus = np.append(self.mus, mus, axis=0)
                if mus.size:
                    self.mu = mus[-1]
            else:
                self.mu = mus
        if Vs is not None:
            if len(Vs.shape) == 4:
                self.Vs = np.append(self.Vs, Vs, axis=0)
                if Vs.size:
                    self.V = Vs[-1]
            else:
                self.V = Vs
        if weightss is not None:
            if len(weightss.shape) == 2:
                self.weightss = np.append(self.weightss, weightss, axis=0)
                if weightss.size:
                    self.weights = weightss[-1]
            else:
                self.weights = weightss
        if logls is not None:
            self.logls = np.append(self.logls, logls)
            if isinstance(logls, (float, int)):
                self.logl = self.logls
            else:
                try:
                    self.logl = self.logls[-1]
                except IndexError:
                    self.logl = None
        if orphanss is not None:
            self.orphanss = np.append(self.orphanss, orphanss)
            if isinstance(orphanss, (float, int)):
                self.orphans = self.orphanss
            else:
                try:
                    self.orphans = self.orphanss[-1]
                except IndexError:
                    self.orphans = None

    def _make_sklearn_gmm(self, n_iter=None):
        n_iter = self.n_iter if n_iter is None else n_iter
        gmm = skl_GMM(self.n_components, max_iter=n_iter, tol=self.tol, reg_covar=self.w,
                covariance_type='full',
                random_state=self.random_state,
                means_init=self.mu,
                precisions_init=None
                )
        if self.V is not None:
            gmm.precisions_init = np.stack([np.linalg.inv(v) for v in self.V])
        if self.weights is not None:
            gmm.weights_init = self.weights / self.weights.sum()
        return gmm

    def _sklearn_fit(self, X, Xcov, Xweights, extra_component_lnprX, no_prints=False):
        tmp_gmm = self._make_sklearn_gmm()
        tmp_gmm.fit(X)
        logl = tmp_gmm.score(X)
        self._append_records(tmp_gmm.means_,  tmp_gmm.covariances_, tmp_gmm.weights_, logl, 0)

    def _xdgmm_fit(self, X, Xcov, Xweights, extra_component_lnprX, validation_X, validation_Xcov,
                   validation_Xweights, no_prints=False):
        if validation_X is None:
            logl = bovyXD(X, Xcov, self.weights, self.mu, self.V, tol=self.tol, maxiter=self.n_iter, w=self.w)
        else:
            bar = tqdm(total=self.n_iter)
            nchunks = 10
            chunksize = self.n_iter // nchunks
            for i in range(0, self.n_iter+chunksize, chunksize):
                bovyXD(X, Xcov, self.weights, self.mu, self.V, tol=self.tol, maxiter=chunksize, w=self.w)
                logl = bovyXD(validation_X, validation_Xcov, self.weights, self.mu, self.V, tol=self.tol, maxiter=chunksize, w=self.w, likeonly=True)
                bar.update(chunksize)
            if self.n_iter % nchunks:
                bovyXD(X, Xcov, self.weights, self.mu, self.V, tol=self.tol, maxiter=self.n_iter % nchunks, w=self.w)
                logl = bovyXD(validation_X, validation_Xcov, self.weights, self.mu, self.V, tol=self.tol, maxiter=chunksize, w=self.w, likeonly=True)
        self._append_records(self.mu, self.V, self.weights, logl, 0)

    def _trunc_fit(self, X, Xcov, Xweights, extra_component_lnprX, validation_X, validation_Xcov, validation_Xweights,
                   no_prints=False, previous_results=None):
        _mus, _Vs, _weightss, _logls, _orphanss = run(self.n_iter, self.mu, self.V, self.weights,
                                                      X, Xcov, self.w,
                                                      self.stdnorm_samples, Xweights, self.learning_rate,
                                                      self.limits, self.fix_components,
                                                      self.groups, extra_component_lnprX,
                                                      self.extra_weight,
                                                      self.tol, self.patience, self.nprocesses,
                                                      validation_X, validation_Xcov, validation_Xweights,
                                                      self.dirichlet_prior,
                                                      no_prints=no_prints,
                                                      previous_data=previous_results)
        self._append_records(_mus, _Vs, _weightss, _logls, _orphanss)

    def initialise(self, X, n_iter=100):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gmm = self._make_sklearn_gmm(n_iter=n_iter).fit(X)
        if self.mu is None:
            self.mu = gmm.means_
        if self.V is None:
            self.V = gmm.covariances_
        if self.weights is None:
            self.weights = gmm.weights_
        self.weights *= (1 - self.extra_weight) / self.weights.sum()

    def fit(self, X, Xcov=None, Xweights=None, extra_component_lnprX=None, n_iter=None, tol=None,
             learning_rate=None, patience=None, checkpoint_every=None,
             validation_X=None, validation_Xcov=None, validation_Xweights=None):
        if n_iter is None:
            n_iter = self.n_iter
        if checkpoint_every is not None:
            assert self.filename is not None
            v, r = divmod(n_iter, checkpoint_every)
            v += r > 0
            n = checkpoint_every
        else:
            v = 1
            n = n_iter
        results = None
        starti = len(self.mus)
        for i in tqdm(range(v), desc='checkpoint', disable=v==1):
            self._fit(X, Xcov, Xweights, extra_component_lnprX, n, tol, learning_rate, patience,
                      validation_X, validation_Xcov, validation_Xweights, no_prints=i > 0,
                      previous_results=results)
            if self.filename:
                self.update_model(self.filename)
            results = (self.mus[starti:], self.Vs[starti:], self.weightss[starti:], self.orphanss[starti:], self.logls[starti:])
        return self


    def _fit(self, X, Xcov=None, Xweights=None, extra_component_lnprX=None, n_iter=None, tol=None,
            learning_rate=None, patience=None,
             validation_X=None, validation_Xcov=None, validation_Xweights=None, no_prints=False,
             previous_results=None):
        assert X.shape[1] == self.ndim
        if Xcov is not None:
            assert Xcov.shape[1:] == (self.ndim, self.ndim)
        old_params = (self.n_iter, self.tol, self.patience, self.learning_rate)
        if n_iter is not None:
            self.n_iter = n_iter
        if tol is not None:
            self.tol = tol
        if patience is not None:
            self.patience = patience
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if self.mu is None or self.V is None or self.weights is None:
            self.initialise(X)
        if self.limits is not None:
            if not no_prints:
                print('Fitting using truncated XDGMM resampling method')
            if Xcov is None:
                self._trunc_fit(X, np.ones((len(X), self.ndim, self.ndim))*X.min()*1e-5,
                                Xweights, extra_component_lnprX,
                                validation_X, validation_Xcov, validation_Xweights, no_prints, previous_results)
            else:
                self._trunc_fit(X, Xcov, Xweights, extra_component_lnprX,
                                validation_X, validation_Xcov, validation_Xweights, no_prints, previous_results)
        else:
            if Xcov is None:
                if not no_prints:
                    print('Fitting using sklearn GMM method')
                if validation_X is not None:
                    raise NotImplementedError
                self._sklearn_fit(X, Xcov, Xweights, extra_component_lnprX, no_prints)
            else:
                if not no_prints:
                    print('Fitting using Bovy XDGMM method')
                self._xdgmm_fit(X, Xcov, Xweights, extra_component_lnprX, validation_X, validation_Xcov, validation_Xweights, no_prints)
        self.n_iter, self.tol, self.patience, self.learning_rate = old_params
        return self

    def score_samples(self, X, Xcov=None, Xweights=None, extra_component_lnprX=None):
        if self.limits is not None:
            if Xcov is None:
                Xcov = np.ones((len(X), self.ndim, self.ndim))*X.min()*1e-5
            qki, pki, total = truncated_qki(self.mu, self.V, self.weights, X, Xcov, self.stdnorm_samples, self.limits,
                                            extra_component_lnprX, Xweights, return_all=True)
            return jlogsumexp(pki, axis=0), np.exp(qki).T
        else:
            if Xcov is None:
                return self._make_sklearn_gmm().score_samples(X)
            else:
                return super().score_samples(X, Xcov)

    def predict(self, X, Xcov, Xweights=None, extra_component_lnprX=None):
        if self.limits is not None:
            if Xcov is None:
                Xcov = np.ones((len(X), self.ndim, self.ndim)) * X.min() * 1e-5
            raise NotImplementedError
        else:
            if Xcov is None:
                return self._make_sklearn_gmm().predict(X)
            else:
                return super().predict(X, Xcov)

    def predict_proba(self, X, Xcov, Xweights=None, extra_component_lnprX=None):
        if self.limits is not None:
            if Xcov is None:
                Xcov = np.ones((len(X), self.ndim, self.ndim)) * X.min() * 1e-5
            raise NotImplementedError
        else:
            if Xcov is None:
                return self._make_sklearn_gmm().predict_proba(X)
            else:
                return super().predict_proba(X, Xcov)

    def logL(self, X, Xcov, Xweights=None, extra_component_lnprX=None):
        if Xcov is None:
            Xcov = np.ones((len(X), self.ndim, self.ndim)) * X.min() * 1e-5
        return jlogsumexp(self.logprob_a(X, Xcov), axis=-1).sum()

    def score(self, X, Xcov, Xweights=None, extra_component_lnprX=None):
        if self.limits is not None:
            if Xcov is None:
                Xcov = np.ones((len(X), self.ndim, self.ndim)) * X.min() * 1e-5
            raise NotImplementedError
        else:
            if Xcov is None:
                return self._make_sklearn_gmm().score(X)
            else:
                return super().score(X, Xcov)

    def logprob_a(self, X, Xcov, Xweights=None):
        if self.limits is not None:
            if Xcov is None:
                Xcov = np.ones((len(X), self.ndim, self.ndim)) * X.min() * 1e-5
            pki = truncated_pki(self.mu, self.V, self.weights, X, Xcov, self.stdnorm_samples,
                                 self.limits, Xweights)
        else:
            if Xcov is None:
                pki = jlog_multivariate_gaussian(X, self.mu[:, None], self.V[:, None]) + jnp.log(self.weights)
            else:
                T = self.V[:, None] + Xcov[None, :]
                pki = jlog_multivariate_gaussian(X, self.mu[:, None], T) + jnp.log(self.weights[:, None])
        return pki.T

    def bic(self, X, Xcov=None, Xweights=None, extra_component_lnprX=None):
        logprob = self.logL(X, Xcov, Xweights, extra_component_lnprX)
        ndim = self.mu.shape[1]
        cov_params = self.n_components * ndim * (ndim + 1) / 2.
        mean_params = ndim * self.n_components
        n_params = int(cov_params + mean_params + self.n_components - 1)
        return (-2 * logprob + n_params * np.log(X.shape[0]))

    def aic(self, X, Xcov=None, Xweights=None, extra_component_lnprX=None):
        logprob = self.logL(X, Xcov, Xweights, extra_component_lnprX)
        ndim = self.mu.shape[1]
        cov_params = self.n_components * ndim * (ndim + 1) / 2.
        mean_params = ndim * self.n_components
        n_params = int(cov_params + mean_params + self.n_components - 1)
        return -2 * logprob + 2 * n_params

    def bic_test(self, component_range, X, Xcov=None, Xweights=None, extra_component_lnprX=None):
        raise NotImplementedError

    def aic_test(self, component_range, X, Xcov=None, Xweights=None, extra_component_lnprX=None):
        raise NotImplementedError

    @property
    def components(self):
        return [self.__class__(1, ndim=self.ndim, n_iter=self.n_iter, tol=self.tol,
                               learning_rate=self.learning_rate, patience=self.patience, labels=self.labels,
                               random_state=self.random_state, n_stdnorm_samples=self.n_stdnorm_samples,
                               mu=[m], V=[v],
                               weights=np.array([1.]), limits=l, extra_weight=self.extra_weight)
                for m, v, l in zip(self.mu, self.V, self.limits)]

    def select_components(self, *ks, renormalise=True):
        try:
            ks = list(ks[0])
        except TypeError:
            pass
        new = deepcopy(self)
        new.n_components = len(ks)
        new.clear_records()
        new.mu, new.V, new.weights = self.mu[ks], self.V[ks], self.weights[ks]
        if self.limits is not None:
            new.limits = self.limits[ks]
        if self.fix_components is not None:
            new.fix_components = {k: v for k, v in self.fix_components.items() if k in ks}
        if self.groups is not None:
            new.groups = [[x for x in g if x in ks] for g in self.groups]
            new.groups = [g for g in new.groups if len(g) > 1]
        if renormalise:
            new.weights /= new.weights.sum()
        return new, self.weights[ks].sum()

    def sample(self, n=1, groups=False, components=False, unweighted=False, ignore_extra_weight=True, random_state=None):
        if not (groups or components):
            if unweighted:
                raise ValueError(f"Sampled a whole model at once cannot be done unweighted")
            if self.limits is None:
                return super().sample(n, random_state)
            else:
                xs = []
                weights = self.weights / self.weights.sum() if ignore_extra_weight else self.weights
                for m, v, l, w in zip(self.mu, self.V, self.limits, weights):
                    x = sample_truncated_gaussian(m, v, l, int(n * w) + 1)
                    xs.append(x)
                _n = n if ignore_extra_weight else int(n * (1 - self.extra_weight))
                xs =  np.concatenate(xs, axis=0)
                if len(xs) > _n:
                    choice = np.random.choice(len(xs), _n, replace=False)
                    return xs[choice]
                return xs
        elif groups:
            if unweighted:
                return [self.select_components(*group)[0].sample(n, False, components) for group in self.groups]
            samples = [self.select_components(*group)[0].sample(int(n * self.weights[group].sum()), False, components)
                    for group in self.groups]
            diff = sum(map(len, samples)) - n
            if diff:
                chosen = np.random.randint(0, len(self.groups), diff)
                samples = [s[:-1] if i in chosen else s for i, s in enumerate(samples)]
            return samples
        else:  # not groups but components
            if unweighted:
                return np.stack([c.sample(n, False, False) for c in self.components])
            samples = [c.sample(int(n * self.weights[c]), False, False) for c in self.components]
            diff = sum(map(len, samples)) - n
            if diff:
                chosen = np.random.randint(0, self.n_components, diff)
                samples = [s[:-1] if i in chosen else s for i, s in enumerate(samples)]
            return [s for s in samples]


    def marginalise(self, ndim):
        raise NotImplementedError