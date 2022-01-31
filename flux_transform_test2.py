from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from functorch import jacrev, vmap
from matplotlib import animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from deconv.gmm.data import DeconvDataset
from deconv.flow.svi import SVIFlow
from nflows.distributions import Distribution
from torch.distributions import MultivariateNormal


def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def jacobianBatch(f, x):
  return vmap(jacrev(f))(x)


sns.set()
SEED = 1234
set_seed(SEED)


# underlying GMM distribution in fitting space
means = torch.Tensor([
    [0.0, 0.0],
    [2, 3],
    [2, -3]
])
covars = torch.Tensor([
    [
        [0.1, 0],
        [0, 1.5]
    ],
    [
        [1, 0],
        [0, 0.1]
    ],
    [
        [1, 0],
        [0, 0.1]
    ]
])

def plane_warp(x):
    x = x.T
    return torch.stack([torch.sinh(x[0]), x[1]]).T


# make a true underlying distribution in data space
# Zfitting = plane_warp(torch.distributions.MultivariateNormal(loc=means, covariance_matrix=covars).sample((5000,)).reshape((-1, 2)))
N = 10_000
Zfitting = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=covars).sample((N,)).reshape((-1, 2))
idx = torch.randperm(Zfitting.shape[0])
Zfitting = Zfitting[idx]


# the first dimension is flux and is transformed to asinh mag to get rid of huge orders of mag range, 2nd dim is left alone
# this functions transform back and forth from data plane (where uncertainties will be gaussians) and the fitting plane
# TODO test normalising data a little better
def data2fitting(x):
    x = x.T
    # return x.T
    return torch.stack([torch.asinh(x[0]), x[1]]).T

def fitting2data(x):
    x = x.T
    # return x.T
    return torch.stack([torch.sinh(x[0]), x[1]]).T

Zdata = fitting2data(Zfitting)

# make uncertainties in data space look
def uncertainty(x):
    """uncertainty covariance of a point x"""
    S = torch.zeros((x.shape[0], x.shape[1], x.shape[1]))
    S[:, 0, 0] = torch.abs(x[:, 0]) * 1  # i.e. scaled poisson noise std=root(flux)
    S[:, 1, 1] = 0.4
    return S


S = uncertainty(Zdata) * 1.  # TODO: test when errors look realistic
noise = torch.distributions.MultivariateNormal(loc=torch.Tensor([0.0, 0.0]), covariance_matrix=S).sample((1,))[0]

# noisy data in both spaces
Xdata = Zdata + noise
Xfitting = data2fitting(Xdata)

fig1, axs = plt.subplots(2, sharex=False, sharey=False)
axs[0].scatter(*Xdata.T, s=1)
axs[0].set_title('noisy data space')
axs[1].scatter(*Xfitting.T, s=1)
axs[1].set_title('noisy fitting space')
# axs[0].set_xlim(None, 100)

fig2, axs = plt.subplots(2, sharex=False, sharey=False)
axs[0].scatter(*Zdata.T, s=1)
axs[0].set_title('True data space')
axs[1].scatter(*Zfitting.T, s=1)
axs[1].set_title('True fitting space')
# axs[0].set_xlim(None, 100)

data_lims = (axs[0].get_xlim(), axs[0].get_ylim())
fitting_lims = (axs[1].get_xlim(), axs[1].get_ylim())


class DeconvGaussianTransformed(Distribution):
    """
    Same as the original but the uncertainty gaussians are uncorrelated.
    Here the uncertainty covariance is assumed to be diagonal, but this is not checked!
    You also specify a transform from fitting space to data space. In this case, the uncertainty
    is assumed to be defined in the data space, not the fitting space.
    """
    def __init__(self, fitting2data, data2fitting):
        super().__init__()
        self.fitting2data = fitting2data
        self.data2fitting = data2fitting

    def log_prob(self, inputs, context):
        X, noise_l = inputs  # fitting space, data space
        # transform samples to the data space where the uncertainties live
        jac = jacobianBatch(self.fitting2data, context)
        context_transformed = self.fitting2data(context)
        log_scaling = vmap(lambda x: torch.slogdet(x)[1])(jac)
        Z = self.fitting2data(X)
        return MultivariateNormal(loc=context_transformed, scale_tril=noise_l).log_prob(Z) + log_scaling


class MySVIFlow(SVIFlow):
    def _create_likelihood(self):
        return DeconvGaussianTransformed(fitting2data, data2fitting)


# bad initialisation occurs if you fit with too few points
X_train = Xfitting[:Xfitting.shape[0] // 2]
S_train = S[:S.shape[0] // 2]
X_test = Xfitting[Xfitting.shape[0] // 2:]
S_test = S[S.shape[0] // 2:]

train_data = DeconvDataset(X_train, torch.cholesky(S_train))
test_data = DeconvDataset(X_test, torch.cholesky(S_test))

fig, axs = plt.subplots(2, 2, sharex='row', sharey='row')
axins_losses = inset_axes(axs[0, 1], width='40%', height='40%', loc='upper right')
axins_losses2 = axins_losses.twinx()
axins_losses2.yaxis.set_visible(True)
axins_losses.set_ylabel('train loss')
axins_losses2.set_ylabel('val loss')

axins_logl = inset_axes(axs[0, 1], width='40%', height='40%', loc='lower right')
axins_logl.set_ylabel('train logl,kl terms')

svi = MySVIFlow(
    2,
    5,
    device= torch.device('cpu'),
    batch_size=2000,
    epochs=2,
    lr=8e-5,
    n_samples=10,
    grad_clip_norm=2,
    use_iwae=True,
)
iterations = enumerate(svi.iter_fit(train_data, test_data, seed=SEED, num_workers=0,
                                    rewind_on_inf=True, return_kl_logl=True))  # iterator

directory = Path('svi_model')
space_dir = directory / 'plots'
loss_dir = directory / 'loss'
params_dir = directory / 'params'
directory.mkdir(parents=True, exist_ok=True)
space_dir.mkdir(parents=True, exist_ok=True)
loss_dir.mkdir(parents=True, exist_ok=True)
params_dir.mkdir(parents=True, exist_ok=True)


# make some test points to visualise the approximate posterior
mean = np.array([[10.0, 0.0], [2.0, 0.0], [5, 3], [5, -3]])  # in fitting space
cov = np.array([  # in data space
    [
        [0.1, 0],
        [0, 3]
    ],
    [
        [0.1, 0],
        [0, 3]
    ],
    [
        [0.5, 0],
        [0, 0.5]
    ],
    [
        [0.5, 0],
        [0, 0.5]
    ]
])

from deconv.gmm.plotting import plot_covariance
test_point = [
    torch.Tensor(mean).to(svi.device),
    vmap(torch.cholesky)(torch.Tensor(cov)).to(svi.device)
]

start_from = 49
if start_from > 0:
    svi.model.load_state_dict(torch.load(params_dir / f'{start_from}.pt'))

train_losses = []
val_losses = []
logls, kls = [], []


def animate_and_save_to_disk(o):
    i, (_svi, train_loss, val_loss, logl, kl) = o
    i += start_from
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    logls.append(logl)
    kls.append(kl)
    torch.save(_svi.model.state_dict(), params_dir / f'{i}.pt')
    Yfitting = _svi.sample_prior(10000)
    Ydata = fitting2data(Yfitting)
    try:
        Qfitting = _svi.resample_posterior(test_point, 10000)  # approximate posterior (`.sample_posterior` would do something different)
    except ValueError:
        print('NaNs in resampling posterior, skipping plotting the approx posterior this iteration')
        Qfitting = None
        Qdata = None
    else:
        Qdata = fitting2data(Qfitting.reshape(-1, Qfitting.shape[-1])).reshape(Qfitting.shape)

    # Draw x and y lists
    for ax in np.ravel(axs):
        ax.clear()
    axs[0, 0].scatter(*Zdata.T.cpu().numpy(), s=1, c='k', label='truth', alpha=0.5)
    axs[0, 0].scatter(*Ydata.T.cpu().numpy(), s=1, label='flow', alpha=0.5)
    axs[0, 0].set_title('model - data space')

    axs[1, 0].scatter(*Zfitting.T.cpu().numpy(), s=1, c='k', label='truth', alpha=0.5)
    axs[1, 0].scatter(*Yfitting.T.cpu().numpy(), s=1, label='flow', alpha=0.5)
    axs[1, 0].set_title('model - fitting space')

    if Qdata is not None:
        for qf, qd, m, c in zip(Qfitting, Qdata, mean, cov):
            axs[1, 1].scatter(*qf.T.cpu().numpy(), s=1)
            pnts = axs[0, 1].scatter(*qd.T.cpu().numpy(), s=1)
            plot_covariance(m, c, ax=axs[0, 1], color=pnts.get_facecolors()[0])
    axs[0, 1].set_title('approx post. - data space')
    axs[1, 1].set_title('approx post. - fitting space')

    axins_losses.clear()
    axins_losses2.clear()
    axins_logl.clear()
    _train_losses = np.array(train_losses)[-20:]
    if len(val_losses):
        _val_losses = np.array(val_losses)[-20:]
    else:
        _val_losses = np.array([])
    if len(_train_losses):
        axins_losses.plot(_train_losses)
    if len(_val_losses):
        axins_losses2.plot(_val_losses)
        try:
            mn, mx = min(_val_losses), max(_val_losses)
            buffer = (mx - mn) * 0.05
            axins_losses2.set_ylim([mn - buffer, mx + buffer])
        except (TypeError, ValueError):
            pass
        try:
            mn, mx = min(_train_losses), max(_train_losses)
            buffer = (mx - mn) * 0.05
            axins_losses.set_ylim([mn - buffer, mx + buffer])
        except (TypeError, ValueError):
            pass

    axins_logl.plot(np.array(logls)[-20:])
    axins_logl.plot(np.array(kls)[-20:])

    axins_losses.set_xscale('log')
    axins_logl.set_xscale('log')
    fig.savefig(space_dir / f'{i}.png')


# run and animate at the same time
# ani = animation.FuncAnimation(fig, animate_and_save_to_disk, interval=500, frames=iterations, repeat=False)
# # run only animate after
for i in iterations:
    pass
animate_and_save_to_disk(i)
plt.show()
