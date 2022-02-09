from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from functorch import jacrev, vmap
from matplotlib import animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from nflows.utils import torchutils

from deconv.gmm.data import DeconvDataset
from deconv.flow.svi import SVIFlow
from nflows.distributions import Distribution
from torch.distributions import MultivariateNormal
torch.autograd.set_detect_anomaly(True)

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
        [0.25, 0],
        [0, 0.1]
    ],
    [
        [0.25, 0],
        [0, 0.1]
    ]
])
N = 10_000

# this is all in data space
Zreal = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=covars).sample((N-(N//5),)).reshape((-1, 2))
# Zreal = torch.where(Zreal > torch.scalar_tensor(10_000, dype=real.dtype), torch.scalar_tensor(10_000, dype=Zreal.dtype), Zreal)
Zreal = torch.clamp(Zreal, -10_000, 10_000)
Zreal[:, 0] = 10**(Zreal[:, 0])  # make it into fluxes
Zfake = torch.distributions.MultivariateNormal(loc=means[:1], covariance_matrix=covars[:1]).sample((N//5,)).reshape((-1, 2))
Zdata = torch.concat([Zreal, Zfake], dim=0)
idx = torch.randperm(Zdata.shape[0])
Zdata = Zdata[idx]


# the first dimension is flux and is transformed to asinh mag to get rid of huge orders of mag range, 2nd dim is left alone
# this functions transform back and forth from data plane (where uncertainties will be gaussians) and the fitting plane
# TODO test normalising data a little better
scaling = 1
cap = 15

def data2fitting(x):
    x = x.T
    # return x.T
    return torch.stack([torch.asinh(x[0]*scaling)/10, x[1]]).T

def fitting2data(x):
    x = x.T
    # return x.T
    return torch.stack([torch.sinh(torch.clamp(x[0] * 10, -cap, cap))/scaling, x[1]]).T

def jac(x):
    return torch.cosh(torch.clamp(x * 10., -cap, cap)) * 10 / scaling


test_val = torch.tensor([[100., 23.4]])
np.testing.assert_allclose(test_val.cpu().numpy(), fitting2data(data2fitting(test_val)).cpu().numpy(), 1e-4)  # float32 is a bit crap

Zfitting = data2fitting(Zdata)

# make uncertainties in data space look
def uncertainty(x):
    """uncertainty covariance of a point x"""
    S = torch.zeros((x.shape[0], x.shape[1], x.shape[1]))
    S[:, 0, 0] = torch.abs(x[:, 0]) * 4  # i.e. scaled poisson noise std=root(flux)
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
        self.data2fitting = data2fitting
        self.fitting2data = fitting2data

    def temporary_use_jacobian(self, x):
        J = torch.zeros((x.shape[0], x.shape[1], x.shape[1]))
        J[:, 0, 0] = jac(x[:, 0])
        J[:, 1, 1] = 1.
        return J

    def log_prob(self, inputs, context):  # TODO: will fail because you can sample very far away from the prior
        X, noise_l = inputs  # fitting space, data space
        # transform samples to the data space where the uncertainties live
        jac = self.temporary_use_jacobian(context)  #jacobianBatch(self.fitting2data, context)
        context_transformed = self.fitting2data(context)
        log_scaling = torch.slogdet(jac)[1]
        Z = self.fitting2data(X)
        context_transformed = torch.where(torch.isfinite(context_transformed), context_transformed,
                                          torch.scalar_tensor(torch.finfo(context_transformed.dtype).min,
                                                              dtype=context_transformed.dtype))
        log_scaling = torch.where(torch.isfinite(log_scaling), log_scaling,
                                          torch.scalar_tensor(torch.finfo(log_scaling.dtype).min,
                                                              dtype=log_scaling.dtype))
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
train_initial_data = DeconvDataset(X_train, torch.cholesky(S_train*0.00001))
test_data = DeconvDataset(X_test, torch.cholesky(S_test))

fig, axs = plt.subplots(2, 2, sharex='row', sharey='row')
axins_losses = inset_axes(axs[0, 1], width='40%', height='40%', loc='upper right')
axins_losses2 = axins_losses.twinx()
axins_losses2.yaxis.set_visible(True)
axins_losses.set_ylabel('train loss')
axins_losses2.set_ylabel('val loss')

axins_logl = inset_axes(axs[0, 1], width='40%', height='40%', loc='lower right')
axins_logl.set_ylabel('train logl,kl terms')

directory = Path('svi_model')
space_dir = directory / 'plots'
loss_dir = directory / 'loss'
params_dir = directory / 'params'
directory.mkdir(parents=True, exist_ok=True)
space_dir.mkdir(parents=True, exist_ok=True)
loss_dir.mkdir(parents=True, exist_ok=True)
params_dir.mkdir(parents=True, exist_ok=True)

total_epochs = 2000
initial_epochs = 0
start_from = 85

svi_initial = SVIFlow(
    2,
    5,
    device= torch.device('cpu'),
    batch_size=2000,
    epochs=initial_epochs,
    lr=1e-6,
    n_samples=10,
    grad_clip_norm=2,
    use_iwae=True,
)

svi = MySVIFlow(
    2,
    5,
    device= torch.device('cpu'),
    batch_size=2000,
    epochs=total_epochs - initial_epochs,
    lr=1e-5,
    n_samples=10,
    grad_clip_norm=2,
    use_iwae=True,
)


if start_from > 0:
    svi_initial.model.load_state_dict(torch.load(params_dir / f'{start_from}.pt'))
    svi.model.load_state_dict(torch.load(params_dir / f'{start_from}.pt'))
    if start_from >= svi_initial.epochs:
        svi_initial.epochs = 0
        svi.epochs -= start_from
    else:
        svi_initial.epochs -= start_from


def iterator():
    i_initial = [start_from]
    for i_initial in enumerate(svi_initial.iter_fit(train_initial_data, seed=SEED, num_workers=0,
                                    rewind_on_inf=True, return_kl_logl=True)):  # iterator
        yield (i_initial[0] + start_from, i_initial[1])
    svi.model.load_state_dict(svi_initial.model.state_dict())  # transfer
    for i in enumerate(svi.iter_fit(train_data, test_data, seed=SEED, num_workers=0,
                                    rewind_on_inf=True, return_kl_logl=True)):  # iterator
        yield (i[0] + i_initial[0], i[1])

iterations = iterator()


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

train_losses = []
val_losses = []
logls, kls = [], []


def animate_and_save_to_disk(o, plot=True):
    i, (_svi, train_loss, val_loss, logl, kl) = o
    i += start_from
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    logls.append(logl)
    kls.append(kl)
    torch.save(_svi.model.state_dict(), params_dir / f'{i}.pt')
    if not plot:
        return
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

    # axins_losses.set_xscale('log')
    # axins_logl.set_xscale('log')
    fig.savefig(space_dir / f'{i}.png')


# run and animate at the same time
ani = animation.FuncAnimation(fig, animate_and_save_to_disk, interval=500, frames=iterations, repeat=False)
# # run only animate after
# for i in iterations:
#     animate_and_save_to_disk(i)
plt.show()
