from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import corner

import seaborn as sns
from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch.autograd.functional import jacobian
from deconv.gmm.data import DeconvDataset
from functorch import jacrev, vmap


def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def jacobianBatch(f, x):
  return vmap(jacrev(f))(x)

# def model_analytic(means, covs, x):
#     torch.distributions.MultivariateNormal(loc=means, covariance_matrix=covars).log_prob(x)
#
# def convolved_posterior_analytic(means, covs, z, zunc):
#

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
Zfitting = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=covars).sample((5000,)).reshape((-1, 2))
idx = torch.randperm(Zfitting.shape[0])
Zfitting = Zfitting[idx]


# the first dimension is flux and is transformed to asinh mag, 2nd dim is left alone
def data2fitting(x):
    x = x.T
    # return x.T
    return torch.stack([torch.asinh(x[0]), x[1]]).T
    # return torch.stack([x[0]*2., x[1]]).T

def fitting2data(x):
    x = x.T
    # return x.T
    return torch.stack([torch.sinh(x[0]), x[1]]).T
    # return torch.stack([x[0]/2., x[1]]).T

Zdata = fitting2data(Zfitting)

# uncertainties in data space
def uncertainty(x):
    """uncertainty covariance of a point x"""
    S = torch.zeros((x.shape[0], x.shape[1], x.shape[1]))
    S[:, 0, 0] = torch.abs(x[:, 0]) * 1  # i.e. scaled poisson noise std=root(flux)
    S[:, 1, 1] = 0.4
    return S
# S = torch.Tensor([
#     [0.1, 0],
#     [0, 0.1]
# ])
S = uncertainty(Zdata)
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

from deconv.flow.svi import SVIFlow
from nflows.distributions import Distribution
from torch.distributions import MultivariateNormal


class DeconvGaussianTransformed(Distribution):
    """
    Same as above but the uncertainty gaussians are uncorrelated.
    Here the uncertainty covariance is assumed to be diagonal, but this is not checked!
    You also specify a transform from fitting space to data space. In this case, the uncertainty
    is assumed to be defined in the data space, not fitting.
    """
    def __init__(self, fitting2data, data2fitting):
        super().__init__()
        self.fitting2data = fitting2data
        self.data2fitting = data2fitting

    def log_prob(self, inputs, context):
        X, noise_l = inputs  # observed data in the data space
        # transform samples to the data space where the uncertainties live
        jac = jacobianBatch(self.fitting2data, context)
        context_transformed = self.fitting2data(context)
        log_scaling = vmap(lambda x: torch.slogdet(x)[1])(jac)
        return MultivariateNormal(loc=context_transformed, scale_tril=noise_l).log_prob(X) + log_scaling


class MySVIFlow(SVIFlow):
    def _create_likelihood(self):
        return DeconvGaussianTransformed(fitting2data, data2fitting)


# bad initialisation occurs if you fit with too few points
svi = MySVIFlow(
    2,
    5,
    device=torch.device('cpu'),
    batch_size=256,
    epochs=3000,
    lr=1e-5,
    n_samples=500,
    use_iwae=True
)

X_train = Xfitting[:Xfitting.shape[0] // 2]
X_test = Xfitting[Xfitting.shape[0] // 2:]
train_data = DeconvDataset(X_train, torch.cholesky(S.repeat(X_train.shape[0], 1, 1)))

fig, axs = plt.subplots(2, 2, sharex='row', sharey='row')
axins = inset_axes(axs[0, 1], width='40%', height='40%')
axins2 = axins.twinx()

iterations = enumerate(svi.iter_fit(train_data, seed=SEED, num_workers=0))  # iterator()


losses = []

directory = Path('svi_model')
space_dir = directory / 'plots'
loss_dir = directory / 'loss'
params_dir = directory / 'params'
directory.mkdir(parents=True, exist_ok=True)
space_dir.mkdir(parents=True, exist_ok=True)
loss_dir.mkdir(parents=True, exist_ok=True)
params_dir.mkdir(parents=True, exist_ok=True)


mean = np.array([[10.0, 0.0], [0.0, 0.0], [30, 3], [30, -3]])
cov = np.array([
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

start_from = 0
if start_from > 0:
    svi.model.load_state_dict(torch.load(params_dir / f'{start_from}.pt'))

def animate_and_record(o):
    i, (_svi, loss) = o
    i += start_from
    losses.append(loss)
    torch.save(_svi.model.state_dict(), params_dir / f'{i}.pt')
    Yfitting = _svi.sample_prior(10000)
    Ydata = fitting2data(Yfitting)
    try:
        Qfitting = _svi.resample_posterior(test_point, 10000)
    except ValueError:
        print('NaNs in resampling posterior, skipping the approx posterior this iteration')
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
    axs[1, 1].set_title('`approx post. - fitting space')

    # axs[1, 1].set_xlim(fitting_lims[0])
    # axs[1, 1].set_ylim(fitting_lims[1])
    #
    # axs[0, 0].set_xlim(data_lims[0])
    # axs[0, 0].set_ylim(data_lims[1])
    #
    # axins.clear()
    # axins.plot(losses)
    # axins.set_ylim([min(losses), max(losses)])
    # axins.set_xscale('log')

    axins.clear()
    axins2.clear()
    _losses = np.array(losses)[-20:]
    # diffs = (_losses[1:] - _losses[:-1]) / np.abs(_losses[:-1])
    if len(_losses):
        axins.plot(_losses)
        try:
            mn, mx = min(_losses), max(_losses)
            buffer = (mx - mn) * 0.05
            axins.set_ylim([mn-buffer, mx+buffer])
        except ValueError:
            pass
    # if len(diffs):
    #     axins2.plot(diffs)
    #     axins2.set_ylim([min(diffs), max(diffs)])
    axins.set_xscale('log')


    fig.savefig(space_dir / f'{i}.png')



# Set up plot to call animate() function periodically
# print(list(svi.model.parameters())[0])
# previous = deepcopy(svi.model.state_dict())
# next(iterations)
# print(list(svi.model.parameters())[0])
# svi.model.load_state_dict(previous)
# print(list(svi.model.parameters())[0])


ani = animation.FuncAnimation(fig, animate_and_record, interval=500, frames=iterations, repeat=False)
# next(iterations)
# next(iterations)
plt.show()
