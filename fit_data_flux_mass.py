from functools import reduce

import astropy.cosmology
from astropy import units as u
from astropy.cosmology import Planck18
from chainconsumer.plotter import Plotter
from matplotlib import rc
import matplotlib.pylab as plt

import numpy as np
import pandas as pd
from pathlib import Path
import math
import seaborn as sns
import torch
from functorch import jacrev, vmap
from matplotlib import animation
from matplotlib.ticker import MaxNLocator, LogLocator, ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from nflows.utils import torchutils

from deconv.flow.nn import DeconvInputEncoder
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


SEED = 1234
set_seed(SEED)


data_file = '~/star-forming/data/sdss-overlapped.csv'
random_model_file = "~/star-forming/data/noise-16comps_seed0_tol1e-08.h5"
random_apertures_file = "~/star-forming/data/merged-random-apertures-radius10.0seed0tagagain2.fits"
random_apertures_flux_name = 'flux'
random_apertures_err_name = "err"
random_apertures_flux_correction = 1/0.17880991378554684
data_flux_correction = 1.
data_flux_name = 'forced_150_flux'
data_err_name = 'forced_150_flux_err'
filter = 'bptclass < 0'
fake_catalogue = False
random_apertures_sane_flux_cutoff = 1000
ncomp_real=1
df = pd.read_csv(data_file).set_index('objid')

df[data_flux_name] *= data_flux_correction
df[data_err_name] *= data_flux_correction

overlapped = df.query(filter)
finite = np.isfinite(overlapped[[data_flux_name, data_err_name]]).all(axis=1) & (overlapped[data_err_name] > 0)
overlapped = overlapped[finite]

def make_luminosity_distance_function(cosmology: astropy.cosmology.FlatLambdaCDM):
    def luminosity_distance(redshift):
        a = 1 / (1 + redshift)



def luminosity(flux, redshift, scale=1.):
    dl = Planck18.luminosity_distance(redshift)
    return (flux * u.mJy * 4 * np.pi * dl**2).to(u.W / u.Hz).value / (1 + redshift)**(-1-0.71) / scale

lum_scale=1e+21
filt = (overlapped['redshift_err'].values / overlapped.redshift.values) < 0.001
print(f"bad redshifts: {sum(~filt)}")

cov = np.zeros([len(overlapped), 3, 3], dtype=np.float32)
cov[:, 0, 0] = luminosity(overlapped[data_err_name].values, overlapped.redshift.values, lum_scale)
cov[:, 1, 1] = (0.434 * overlapped['redshift_err'].values / overlapped.redshift.values) ** 2.
cov[:, 2, 2] = ((overlapped['lgm_tot_p84'] - overlapped['lgm_tot_p16']) / 2).values ** 2.
cov = torch.tensor(cov[filt])
Xdata = torch.tensor(np.stack([
    luminosity(overlapped[data_flux_name].values, overlapped.redshift.values, lum_scale),
    np.log10(overlapped.redshift.values + 1),
    overlapped.lgm_tot_p50.values]).T[filt].astype(np.float32))


# TODO test normalising data a little better
scaling = 100
cap = 15

def data2fitting(x):
    x = x.T
    return torch.stack([
        torch.asinh(x[0]*scaling),
        x[1] - 0.03,
        x[2] - 10.5
    ]).T

def fitting2data(x):
    x = x.T
    return torch.stack([
        torch.sinh(torch.clamp(x[0], -cap, cap))/scaling,
        x[1] + 0.03,
        x[2] + 10.5
    ]).T

def jac(x):
    return torch.cosh(torch.clamp(x * 10., -cap, cap)) * 10 / scaling


import corner
from chainconsumer import ChainConsumer


Xfitting = data2fitting(Xdata)

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
        J[:, 2, 2] = 1.#10**x[:, 1] * torch.log(torch.scalar_tensor(10., dtype=J.dtype))
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


class MyPlotter(Plotter):
    fig_data = None

    def _get_figure(self, all_parameters, flip, figsize=(5, 5), external_extents=None, chains=None, blind=None, log_scales=None):
        n = len(all_parameters)
        max_ticks = self.parent.config["max_ticks"]
        spacing = self.parent.config["spacing"]
        plot_hists = self.parent.config["plot_hists"]
        label_font_size = self.parent.config["label_font_size"]
        tick_font_size = self.parent.config["tick_font_size"]
        diagonal_tick_labels = self.parent.config["diagonal_tick_labels"]
        if blind is None:
            blind = []

        if chains is None:
            chains = self.parent.chains

        if not plot_hists:
            n -= 1

        if spacing is None:
            spacing = 1.0 if n < 6 else 0.0

        if n == 2 and plot_hists and flip:
            gridspec_kw = {"width_ratios": [3, 1], "height_ratios": [1, 3]}
        else:
            gridspec_kw = {}

        if self.fig_data is None:
            fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False, gridspec_kw=gridspec_kw)
            self.fig_data = fig, axes
        else:
            fig, axes = self.fig_data
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.05 * spacing, hspace=0.05 * spacing)

        extents = self._get_custom_extents(all_parameters, chains, external_extents)

        if plot_hists:
            params1 = all_parameters
            params2 = all_parameters
        else:
            params1 = all_parameters[1:]
            params2 = all_parameters[:-1]
        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                ax = axes[i, j]
                formatter_x = ScalarFormatter(useOffset=True)
                formatter_x.set_powerlimits((-3, 4))
                formatter_y = ScalarFormatter(useOffset=True)
                formatter_y.set_powerlimits((-3, 4))

                display_x_ticks = False
                display_y_ticks = False
                if i < j:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    logx = False
                    logy = False
                    if p1 == p2:
                        if log_scales.get(p1):
                            if flip and j == n - 1:
                                ax.set_yscale("log")
                                logy = True
                            else:
                                ax.set_xscale("log")
                                logx = True
                    else:
                        if log_scales.get(p1):
                            ax.set_yscale("log")
                            logy = True
                        if log_scales.get(p2):
                            ax.set_xscale("log")
                            logx = True
                    if i != n - 1 or (flip and j == n - 1):
                        ax.set_xticks([])
                    else:
                        if p2 in blind:
                            ax.set_xticks([])
                        else:
                            display_x_ticks = True
                        if isinstance(p2, str):
                            ax.set_xlabel(p2, fontsize=label_font_size)
                    if j != 0 or (plot_hists and i == 0):
                        ax.set_yticks([])
                    else:
                        if p1 in blind:
                            ax.set_yticks([])
                        else:
                            display_y_ticks = True
                        if isinstance(p1, str):
                            ax.set_ylabel(p1, fontsize=label_font_size)
                    if display_x_ticks:
                        if diagonal_tick_labels:
                            _ = [l.set_rotation(45) for l in ax.get_xticklabels()]
                        _ = [l.set_fontsize(tick_font_size) for l in ax.get_xticklabels()]
                        if not logx:
                            ax.xaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                            ax.xaxis.set_major_formatter(formatter_x)
                        else:
                            ax.xaxis.set_major_locator(LogLocator(numticks=max_ticks))
                    else:
                        ax.set_xticks([])
                    if display_y_ticks:
                        if diagonal_tick_labels:
                            _ = [l.set_rotation(45) for l in ax.get_yticklabels()]
                        _ = [l.set_fontsize(tick_font_size) for l in ax.get_yticklabels()]
                        if not logy:
                            ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                            ax.yaxis.set_major_formatter(formatter_y)
                        else:
                            ax.yaxis.set_major_locator(LogLocator(numticks=max_ticks))
                    else:
                        ax.set_yticks([])
                    if i != j or not plot_hists:
                        ax.set_ylim(extents[p1])
                    elif flip and i == 1:
                        ax.set_ylim(extents[p1])
                    ax.set_xlim(extents[p2])

        return fig, axes, params1, params2, extents

class MyChainConsumer(ChainConsumer):
    def __init__(self):
        super().__init__()
        self.plotter = MyPlotter(self)


def plot_samples(x_data, x_fitting, y_data, y_fitting, c=None):
    if c is None:
        c = MyChainConsumer()
    observed = torch.concat([x_data, x_fitting], axis=1).numpy()
    fit = torch.concat([y_data, y_fitting], axis=1).numpy()

    for i in range(len(c.chains)):
        c.remove_chain()
    perc = [0.0001, 99.]
    extents = [(np.percentile(x, perc), np.percentile(y, perc)) for x, y in zip(observed.T, fit.T)]
    extents = [(min([x[0], y[0]]), max([x[1], y[1]])) for x, y in extents]
    filt = reduce(lambda a, b: a & b, ((o > l) & (o < u) for o, (l, u) in zip(observed.T, extents)))
    c.add_chain(observed[filt],
                parameters=['L150', 'log(1+z)', 'logM']+['asinh(L150)', 'log(1+z)-0.03', 'logM-10.5'],
                name='observed',
                kde=False)
    filt = reduce(lambda a, b: a & b, ((o > l) & (o < u) for o, (l, u) in zip(fit.T, extents)))
    c.add_chain(fit[filt],
                parameters=['L150', 'log(1+z)', 'logM']+['asinh(L150)', 'log(1+z)-0.03', 'logM-10.5'],
                name='fit',
                kde=False)
    c.configure(usetex=False, smooth=0, cloud=True, shade_alpha=0.15)
    c.plotter.plot()
    return c

S = cov
partition = 3
X_test = Xfitting[:Xfitting.shape[0] // partition]
S_test = S[:S.shape[0] // partition]
X_train = Xfitting[Xfitting.shape[0] // partition:]
S_train = S[S.shape[0] // partition:]

train_data = DeconvDataset(X_train, torch.cholesky(S_train))
test_data = DeconvDataset(X_test, torch.cholesky(S_test))

directory = Path('svi_model')
space_dir = directory / 'plots'
loss_dir = directory / 'loss'
params_dir = directory / 'params'
directory.mkdir(parents=True, exist_ok=True)
space_dir.mkdir(parents=True, exist_ok=True)
loss_dir.mkdir(parents=True, exist_ok=True)
params_dir.mkdir(parents=True, exist_ok=True)

total_epochs = 10_000
start_from = 0
svi = MySVIFlow(
    3,
    5,
    device= torch.device('cpu'),
    batch_size=2000,
    epochs=total_epochs,
    lr=1e-4,
    n_samples=10,
    grad_clip_norm=2,
    use_iwae=True,
)

if start_from > 0:
    svi.model.load_state_dict(torch.load(params_dir / f'{start_from}.pt'))
    svi.epochs -= start_from

def iterator():
    for i in enumerate(svi.iter_fit(train_data, test_data, seed=SEED, num_workers=0,
                                    rewind_on_inf=True, return_kl_logl=True)):  # iterator
        yield (i[0] + start_from, i[1])

iterations = iterator()


train_losses = []
val_losses = []
logls, kls = [], []
c = plot_samples(Xdata, Xfitting, Xdata, Xfitting)

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
    Yfitting = _svi.sample_prior(10000)[0]
    Ydata = fitting2data(Yfitting)
    for ax in c.plotter.fig_data[0].axes:
        ax.clear()
    plot_samples(Xdata, Xfitting, Ydata, Yfitting, c)
    c.plotter.fig_data[0].savefig(space_dir / f'{i}.png')

# fig = plt.figure()
# ani = animation.FuncAnimation(fig, animate_and_save_to_disk, interval=500, frames=iterations, repeat=False)
ani = animation.FuncAnimation(c.plotter.fig_data[0], animate_and_save_to_disk, interval=500, frames=iterations, repeat=False)
# consumer = plot_samples(Xdata, Xfitting, Xdata, Xfitting)
# for ax in consumer.plotter.fig_data[0].axes:
#     ax.clear()
# plot_samples(Xdata, Xfitting, Xdata+0.01, Xfitting+0.01, consumer)
plt.show()