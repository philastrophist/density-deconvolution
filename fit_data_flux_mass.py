import logging
import warnings
from collections import namedtuple
from functools import reduce
from pathlib import Path
from typing import Any

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import torch
from astropy import units as u
from astropy.cosmology import Planck18
from chainconsumer.plotter import Plotter
from functorch import jacrev, vmap
from matplotlib import animation
from matplotlib.ticker import MaxNLocator, LogLocator, ScalarFormatter
from nflows.distributions import Distribution
from scipy import optimize
from torch.autograd import Function
from torch.distributions import MultivariateNormal

from deconv.flow.svi import SVIFlow
from deconv.gmm.data import DeconvDataset
from differentiation import LocDifferentiableMultivariateNormal

logging.getLogger('chainconsumer').setLevel(logging.CRITICAL)

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
fake_catalogue = False
random_apertures_sane_flux_cutoff = 1000
ncomp_real=1
df = pd.read_csv(data_file).set_index('objid')

df[data_flux_name] *= data_flux_correction
df[data_err_name] *= data_flux_correction

# filter = 'bptclass < 0'
overlapped = df#.query(filter)
finite = np.isfinite(overlapped[[data_flux_name, data_err_name]]).all(axis=1) & (overlapped[data_err_name] > 0)
overlapped = overlapped[finite].sample(frac=1, random_state=SEED)


def luminosity(flux, redshift, scale=1.):
    dl = Planck18.luminosity_distance(redshift)
    return (flux * u.mJy * 4 * np.pi * dl**2).to(u.W / u.Hz).value / (1 + redshift)**(-1-0.71) / scale


lum_scale=1e+21
line_scale = 1000.
redshift_filt = (overlapped['redshift_err'].values / overlapped.redshift.values) < 0.001
redshift_filt &= overlapped.zwarning == 0
print(f"bad redshifts: {sum(~redshift_filt)}")

lines = 'h_alpha_flux h_beta_flux nii_6584_flux oiii_5007_flux'.split()
line_filt = np.ones_like(redshift_filt, dtype=bool)
for line in lines:
    line_filt &= (np.abs(overlapped[line]) < 1e+4) & (overlapped[line+'_err'] > 0) & np.isfinite(overlapped[line+'_err'])
print(f"outlier line fluxes: {sum(~line_filt)}")
overlapped = overlapped[redshift_filt & line_filt]
print(f"using {len(overlapped)} data points")
# data dimensions to use:
# L150
# redshift
# M*
# ha, hb, nii, oiii
Xdata = torch.tensor(np.stack([
    luminosity(overlapped[data_flux_name].values, overlapped.redshift.values, lum_scale),
    overlapped.redshift.values,
    overlapped.lgm_tot_p50.values,
    overlapped.h_alpha_flux.values / line_scale,
    overlapped.h_beta_flux.values / line_scale,
    overlapped.nii_6584_flux.values / line_scale,
    overlapped.oiii_5007_flux.values / line_scale,
]).T.astype(np.float32))
cov = np.zeros([len(overlapped), 7, 7], dtype=np.float32)
cov[:, 0, 0] = luminosity(overlapped[data_err_name].values, overlapped.redshift.values, lum_scale)
cov[:, 1, 1] = overlapped['redshift_err'].values
cov[:, 2, 2] = ((overlapped['lgm_tot_p84'] - overlapped['lgm_tot_p16']) / 2).values
cov[:, 3, 3] = overlapped['h_alpha_flux_err'].values / line_scale
cov[:, 4, 4] = overlapped['h_beta_flux_err'].values / line_scale
cov[:, 5, 5] = overlapped['nii_6584_flux_err'].values / line_scale
cov[:, 6, 6] = overlapped['oiii_5007_flux_err'].values / line_scale
cov = torch.tensor(cov)**2.


# TODO test normalising data a little better
scaling = 100
cap = 20.


def inv_logit(x):
    y = torch.exp(x)
    return y / (1 + y)

np.testing.assert_allclose(inv_logit(torch.logit(torch.scalar_tensor(0.3425))).numpy(), 0.3425)

TransformPair = namedtuple('TransformPair', ['data2fitting', 'fitting2data', 'jacobian'])

def make_hard_bounder(x, base_transform=lambda x: x, buffer=0.001):
    """
    Returns the transformation function pair which makes a logit bound on x based on the data spread
    i.e. the model is bounded to the spread of the data plus a little bit of a buffer
    """
    bounds = torch.min(x), torch.max(x)
    width = bounds[1] - bounds[0]
    # add buffer to avoid infs
    bounds = bounds[0] - width * buffer, bounds[1] + width * buffer
    width = bounds[1] - bounds[0]

    def rescale(y):
        """
        transform y to the [0, 1] region
        """
        return (y - bounds[0]) / width

    def inv_rescale(y):
        """
        transform y back to its original boundaries
        """
        return (y * width) + bounds[0]

    def _data2fitting(y):
        return torch.logit(rescale(y))

    def _fitting2data(y):
        return inv_rescale(inv_logit(y))

    def _jacobian(y):
        exp = torch.exp(y)
        return width * exp / ((exp + 1)**2)

    return TransformPair(_data2fitting, _fitting2data, _jacobian)

def ztransform(x, transform, data):
    p = transform(data)
    return (transform(x) - p.mean()) / p.std()

def inv_ztransform(x, transform, data):
    p = transform(data)
    return (x * p.std()) + p.mean()

def jac_ztransform_multiplier(transform, data):
    p = transform(data)
    return p.std()

# def generalised_sigmoid(x, lower=0., upper=1., zero_value=0.5, shift=0., loggrowth=0., logv=0., c=1.):
#     growth = np.exp(loggrowth)
#     v = np.exp(logv)
#     a = lower
#     k = ((upper - a) * (c ** (1/v))) + a
#     q = (((k - a) / (zero_value - a))**v) - c
#     exp = torch.exp(-growth * (x - shift)) * q
#     return a + ((k - a) / (c + exp))**(1/v)
#
# def switch(x, lower, upper, lower_value, upper_value):
#     grad = (upper_value - lower_value) / (upper - lower)
#     inter = upper_value - (grad * upper)
#     return torch.where(x >= upper, upper_value, torch.where(x <= lower, lower_value, inter + (grad * x)))
#
#
# def fit_soft(lower, upper, lower_value, upper_value, npoints=1000):
#     buffer = (upper - lower) * 0.5
#     x = np.linspace(lower-buffer, upper+buffer, npoints)
#     y = switch(torch.as_tensor(x), lower, upper, lower_value, upper_value).numpy()
#     mid = (upper - lower) / 2.
#     mid_value = (upper_value - lower_value) / 2.
#     p0 = [0., 0., 1.]
#     popt, _ = optimize.curve_fit(lambda xx, *p:
#                                  generalised_sigmoid(torch.as_tensor(xx), lower, upper, mid_value, mid, *p).numpy(), x, y, p0)
#     return popt
#

# def soft_boundary(x, lower, upper, lower_value, upper_value, boundary_width):
#     l = torch.sigmoid(-(x-lower) / boundary_width)
#     u = torch.sigmoid((x-upper) / boundary_width)
#     centre = (1-l) * (1-u)
#     grad = (upper_value - lower_value) / (upper - lower)
#     inter = upper_value - (grad * upper)
#     y = (grad * x) + inter
#     return (lower_value * l) + (upper_value * u) + (centre * y)
redshift_bounder = make_hard_bounder(Xdata[:, 1])

def data2fitting(x):
    x = x.T
    return torch.stack([
        torch.asinh(x[0] * scaling),
        redshift_bounder.data2fitting(x[1]),
        x[2] - 10.5,
        ztransform(x[3], torch.asinh, Xdata[:, 3]),
        ztransform(x[4], torch.asinh, Xdata[:, 4]),
        ztransform(x[5], torch.asinh, Xdata[:, 5]),
        ztransform(x[6], torch.asinh, Xdata[:, 6])
    ]).T
data2fitting.names = ['asinh(scaled_L150)', 'logit(z)', 'logM-10.5', 'asinh(ha)', 'asinh(hb)', 'asinh(nii)', 'asinh(oiii)']

def fitting2data(x):
    x = x.T
    return torch.stack([
        torch.sinh(torch.clamp(x[0], -cap, cap)) / scaling,
        redshift_bounder.fitting2data(torch.clamp(x[1], -10., 10.)),
        x[2] + 10.5,
        torch.sinh(torch.clamp(inv_ztransform(x[3], torch.asinh, Xdata[:, 3]), -cap, cap)),
        torch.sinh(torch.clamp(inv_ztransform(x[4], torch.asinh, Xdata[:, 4]), -cap, cap)),
        torch.sinh(torch.clamp(inv_ztransform(x[5], torch.asinh, Xdata[:, 5]), -cap, cap)),
        torch.sinh(torch.clamp(inv_ztransform(x[6], torch.asinh, Xdata[:, 6]), -cap, cap)),
    ]).T
fitting2data.names = ['scaled_L150', 'z', 'logM', 'ha', 'hb', 'nii', 'oiii']

def jacobian(x):
    J = torch.zeros((x.shape[0], x.shape[1], x.shape[1]))
    J[:, 0, 0] = torch.cosh(torch.clamp(x[:, 0], -cap, cap)) / scaling
    J[:, 1, 1] = redshift_bounder.jacobian(torch.clamp(x[:, 1], -10., 10.))
    J[:, 2, 2] = 1.
    J[:, 3, 3] = torch.cosh(torch.clamp(x[:, 3], -cap, cap)) * jac_ztransform_multiplier(torch.asinh, Xdata[:, 3])
    J[:, 4, 4] = torch.cosh(torch.clamp(x[:, 4], -cap, cap)) * jac_ztransform_multiplier(torch.asinh, Xdata[:, 4])
    J[:, 5, 5] = torch.cosh(torch.clamp(x[:, 5], -cap, cap)) * jac_ztransform_multiplier(torch.asinh, Xdata[:, 5])
    J[:, 6, 6] = torch.cosh(torch.clamp(x[:, 6], -cap, cap)) * jac_ztransform_multiplier(torch.asinh, Xdata[:, 6])
    return J

def data2auxillary(x):
    """
    Adds other dimension that may be used for plotting/diagnostics etc
    Not used in the actual fit
    """
    x = x.T
    return torch.stack([
        torch.log10(x[0]*lum_scale),
        torch.log10(x[-2]) - torch.log10(x[-4]),  # line_scale is not needed since it's a ratio
        torch.log10(x[-1]) - torch.log10(x[-3])
        ]).T
data2auxillary.names = ['log(L150)', 'log(nii / Ha)', 'log(oiii / Hb)']


from chainconsumer import ChainConsumer


Xfitting = data2fitting(Xdata)
# np.testing.assert_allclose(fitting2data(Xfitting), Xdata, 2e-6)

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

    def log_prob(self, inputs, context):
        X, noise_l = inputs  # fitting space, data space
        # transform samples to the data space where the uncertainties live
        jac = jacobian(context)  #jacobianBatch(self.fitting2data, context_fittingspace)
        context_dataspace = self.fitting2data(context)
        log_scaling = torch.slogdet(jac)[1]
        Z = self.fitting2data(X)
        context_dataspace = torch.where(torch.isfinite(context_dataspace), context_dataspace,
                                          torch.scalar_tensor(torch.finfo(context_dataspace.dtype).min,
                                                              dtype=context_dataspace.dtype))
        log_scaling = torch.where(torch.isfinite(log_scaling), log_scaling,
                                          torch.scalar_tensor(torch.finfo(log_scaling.dtype).min,
                                                              dtype=log_scaling.dtype))
        return LocDifferentiableMultivariateNormal(loc=context_dataspace, scale_tril=noise_l).log_prob(Z) + log_scaling


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


def plot_samples(x_data, x_fitting, x_aux, y_data, y_fitting, y_aux, c=None):
    if c is None:
        c = MyChainConsumer()
    observed = torch.concat([x_data, x_fitting, x_aux], axis=1).numpy()
    fit = torch.concat([y_data, y_fitting, y_aux], axis=1).numpy()

    for i in range(len(c.chains)):
        c.remove_chain()
    perc = [0.0001, 99.]
    extents = [(np.nanpercentile(x[np.isfinite(x)], perc), np.nanpercentile(y[np.isfinite(y)], perc)) for x, y in zip(observed.T, fit.T)]
    extents = [(min([x[0], y[0]]), max([x[1], y[1]])) for x, y in extents]
    filt = reduce(lambda a, b: a & b, ((o > l) & (o < u) for o, (l, u) in zip(observed.T, extents)))
    c.add_chain(observed[filt],
                parameters=fitting2data.names+data2fitting.names+data2auxillary.names,
                name='observed',
                kde=False)
    filt = reduce(lambda a, b: a & b, ((o > l) & (o < u) for o, (l, u) in zip(fit.T, extents)))
    c.add_chain(fit[filt],
                parameters=fitting2data.names+data2fitting.names+data2auxillary.names,
                name='fit',
                kde=False)
    c.configure(usetex=False, smooth=0, cloud=True, shade_alpha=0.15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        c.plotter.plot()
    return c

S = cov
partition = 10
X_test = Xfitting[:Xfitting.shape[Xdata0] // partition]
S_test = S[:S.shape[0] // partition]
X_train = Xfitting[Xfitting.shape[0] // partition:]
S_train = S[S.shape[0] // partition:]
print(f"validation={len(X_test)}, training={len(X_train)} ({1 / partition:.2%})")

train_data = DeconvDataset(X_train, torch.linalg.cholesky(S_train))
test_data = DeconvDataset(X_test, torch.linalg.cholesky(S_test))

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
plot_every = 10
svi = MySVIFlow(
    Xdata.shape[1],
    flow_steps=5,
    device= torch.device('cpu'),
    batch_size=2000,
    epochs=total_epochs,
    lr=1e-3,
    n_samples=10,
    grad_clip_norm=2,
    use_iwae=True,
)

if start_from > 0:
    svi.model.load_state_dict(torch.load(params_dir / f'{start_from}.pt'))
    svi.epochs -= start_from

def iterator():
    for i in enumerate(svi.iter_fit(train_data, test_data, seed=SEED, num_workers=8,
                                    rewind_on_inf=True, return_kl_logl=True), start_from):  # iterator
        yield i


iterations = iterator()

epochs = []
train_losses = []
val_losses = []
logls, kls = [], []
Yfitting = svi.sample_prior(10000)[0]
Ydata = fitting2data(Yfitting)
Yaux = data2auxillary(Ydata)
Xaux = data2auxillary(Xdata)
c = plot_samples(Xdata, Xfitting, Xaux, Ydata, Yfitting, Yaux)
axins_losses = c.plotter.fig_data[0].add_axes([0.6, 0.6, 0.3, 0.3])

def autoscale_y(ax,margin=0.05):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        if len(y_displayed):
            h = np.max(y_displayed) - np.min(y_displayed)
            bot = np.min(y_displayed)-margin*h
            top = np.max(y_displayed)+margin*h
            return bot,top
        return lo, hi

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)

def animate_and_save_to_disk(o):
    i, (_svi, train_loss, val_loss, logl, kl) = o
    epochs.append(i)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    logls.append(logl)
    kls.append(kl)
    torch.save(_svi.model.state_dict(), params_dir / f'{i}.pt')
    Yfitting = _svi.sample_prior(10000)[0]
    Ydata = fitting2data(Yfitting)
    if i % plot_every:
        return

    for ax in c.plotter.fig_data[0].axes:
        ax.clear()
    Yaux = data2auxillary(Ydata)
    Xaux = data2auxillary(Xdata)
    plot_samples(Xdata, Xfitting, Xaux, Ydata, Yfitting, Yaux, c)
    axins_losses.clear()
    if len(train_losses):
        axins_losses.plot(epochs, train_losses, label='train')
    if len(val_losses):
        axins_losses.plot(epochs, val_losses, label='val')
    axins_losses.set_xlim(max([0, max(epochs) - 20]), max(epochs))
    autoscale_y(axins_losses)
    axins_losses.legend(loc='lower right')
    plt.suptitle(f'epoch {i}')
    fig = c.plotter.fig_data[0]
    fig.savefig(space_dir / f'{i}.png')

ani = animation.FuncAnimation(c.plotter.fig_data[0], animate_and_save_to_disk, interval=500, frames=iterations, repeat=False)
plt.show()
# for _ in iterations:
#     pass