import logging
import warnings
from collections import namedtuple
from functools import reduce
from pathlib import Path
from typing import Any

import matplotlib.pylab as plt
import nflows.transforms
import numpy as np
import pandas as pd
import torch
from astropy import units as u
from astropy.cosmology import Planck18
from astropy.table import Table
from chainconsumer.helpers import get_grid_bins, get_smoothed_bins
from chainconsumer.plotter import Plotter
from functorch import jacrev, vmap
from matplotlib import animation
from matplotlib.ticker import MaxNLocator, LogLocator, ScalarFormatter
from nflows.distributions import Distribution
from nflows import utils
from nflows.transforms import Sigmoid
from scipy import optimize
from scipy.interpolate import interp1d
from torch.autograd import Function
from torch.distributions import MultivariateNormal, Normal
from tqdm import tqdm

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

from custom_xdgmm import GeneralisedXDGMM as XDGMM


def jacobianBatch(f, x):
  return vmap(jacrev(f))(x)


def random_catalogue_match_fields(real_catalogue, random_fluxes):
    """
    1. value_count field
    2. extract 1 random flux for each occurance of each field
    3. assign to real_catalogue
    """
    randomised = pd.DataFrame(index=real_catalogue.index, columns=['flux', 'err'])
    counts = real_catalogue.field.value_counts()
    selected = []
    for field, count in tqdm(counts.items(), total=len(counts)):
        subset = random_fluxes[random_fluxes.field == field]
        choice = np.random.randint(0, len(subset), count)
        subset = subset.iloc[choice]
        randomised.loc[real_catalogue.field == field, 'flux'] = subset['flux'].values
        randomised.loc[real_catalogue.field == field, 'err'] = subset['err'].values
        selected.append(subset.index.values)
    return randomised.applymap(float), np.concatenate(selected)


SEED = 12345
set_seed(SEED)


data_file = '~/star-forming/data/sdss-magnitudes-joined.csv'
random_model_file = "~/star-forming/data/noise-16comps_seed0_tol1e-08.h5"
random_apertures_file = "~/star-forming/data/merged-random-apertures-radius10.0seed0tagagain2.fits"
random_ncomps = 16
random_apertures_flux_name = 'flux'
random_apertures_err_name = "err"
random_apertures_flux_correction = 1/0.17880991378554684
data_flux_correction = 1.
data_flux_name = 'forced_150_flux'
data_err_name = 'forced_150_flux_err'
fake_catalogue = False
random_apertures_sane_flux_cutoff = 0.2
df = pd.read_csv(data_file).set_index('objid')

df[data_flux_name] *= data_flux_correction
df[data_err_name] *= data_flux_correction

# filter = 'bptclass < 0'
overlapped = df#.query(filter)
finite = np.isfinite(overlapped[[data_flux_name, data_err_name]]).all(axis=1) & (overlapped[data_err_name] > 0)
overlapped = overlapped[finite].sample(frac=1, random_state=SEED)

# now read in the random apertures
if not Path(random_model_file).expanduser().exists():
    random = Table.read(Path(random_apertures_file).expanduser()).to_pandas()
    try:
        random['field'] = random['field'].apply(lambda x: x.decode())
    except AttributeError:
        pass
    random[random_apertures_flux_name] *= random_apertures_flux_correction
    random[random_apertures_err_name] *= random_apertures_flux_correction
    random[random_apertures_flux_name] = random[random_apertures_flux_name].apply(float)
    random[random_apertures_err_name] = random[random_apertures_err_name].apply(float)
    random = random[random[random_apertures_flux_name] < random_apertures_sane_flux_cutoff]

    randomised, selected_indexes = random_catalogue_match_fields(overlapped, random)
    if fake_catalogue:
        # make sure we don't get the same random sample
        fake = random_catalogue_match_fields(overlapped, random[~random.isin(selected_indexes)])[0]
        overlapped[data_flux_name] = fake[random_apertures_flux_name]
        overlapped[data_err_name] = fake[random_apertures_err_name]

    cov_random = np.zeros([len(random), 1, 1])
    cov_random[:, 0, 0] = random[random_apertures_err_name].values ** 2.
    X_random = random[random_apertures_flux_name].values[:, None]

    random_xdgmm = XDGMM(random_ncomps, n_iter=1000)
    random_xdgmm.fit(X_random, cov_random)
else:
    random_xdgmm = XDGMM.from_file(Path(random_model_file).expanduser())

def luminosity(flux, redshift, scale=1.):
    dl = Planck18.luminosity_distance(redshift)
    return (flux * u.mJy * 4 * np.pi * dl**2).to(u.W / u.Hz).value / (1 + redshift)**(-1-0.71) / scale

def set_large_uncertainty(table, name, err_name, perc_width=99., multipler=2, other_filt=None, ivar=False):
    filt = (table[err_name] <= 0) | ~np.isfinite(table[err_name]) | ~np.isfinite(table[name])
    if other_filt is not None:
        filt &= other_filt
    if sum(filt):
        print(f'setting {sum(filt)} {err_name} to {multipler} * {perc_width}%')
        table[name][filt] = table[name][~filt].mean()
        width = np.percentile(table[name][~filt], (100-perc_width, perc_width))
        x = (width[1] - width[0]) * multipler
        if ivar:
            x = 1 / x**2
        table[err_name][filt] = x


lum_scale=1e+21
line_scale = 1000.
redshift_filt = (overlapped['redshift_err'].values / overlapped.redshift.values) < 0.001
redshift_filt &= overlapped.zwarning == 0
print(f"bad redshifts: {sum(~redshift_filt)}")

line_filt = np.ones(len(overlapped), dtype=bool)
lines = 'h_alpha_flux h_beta_flux nii_6584_flux oiii_5007_flux'.split()
for line in lines:
    line_filt &= np.abs(overlapped[line]) < 1e+4
    line_filt &= (overlapped[line+'_err'] > 0) & np.isfinite(overlapped[line+'_err'])
print(f"outlier line fluxes: {sum(~line_filt)}")
overlapped = overlapped[redshift_filt & line_filt]
for line in lines:
    set_large_uncertainty(overlapped, line, line+'_err')  # set unknown to large variance
set_large_uncertainty(overlapped, 'D4000', 'D4000_ERR')
overlapped['lgm_tot_p50_err'] = (overlapped['lgm_tot_p84'] - overlapped['lgm_tot_p16']) / 2
set_large_uncertainty(overlapped, 'lgm_tot_p50', 'lgm_tot_p50_err')
set_large_uncertainty(overlapped, 'cModelFlux_r', 'cModelFluxIvar_r', ivar=True)

lumdist = Planck18.luminosity_distance(overlapped['redshift'])
distmod = Planck18.distmod(overlapped['redshift']).value
overlapped['cModelMag_r'] = 22.5 - 2.5 * np.log10(overlapped['cModelFlux_r'])  - distmod
overlapped['cModelMagErr_r'] = np.abs(2.5 * 0.434 / np.sqrt(overlapped['cModelFluxIvar_r']) / overlapped['cModelFlux_r'])

radio_flux_scaling = 1000.

Xdata = torch.tensor(np.stack([
    overlapped[data_flux_name] * radio_flux_scaling,
    overlapped.redshift.values,
    overlapped.lgm_tot_p50.values,
    overlapped.h_alpha_flux.values,
    overlapped.h_beta_flux.values,
    overlapped.nii_6584_flux.values,
    overlapped.oiii_5007_flux.values,
    overlapped.D4000.values,
    # overlapped.cModelFlux_r.values,
    overlapped.cModelMag_r.values,

]).T.astype(np.float32))
Xerr = torch.tensor(np.stack([
    overlapped[data_err_name] * radio_flux_scaling,
    overlapped['redshift_err'].values,
    ((overlapped['lgm_tot_p84'] - overlapped['lgm_tot_p16']) / 2).values,
    overlapped['h_alpha_flux_err'].values,
    overlapped['h_beta_flux_err'].values,
    overlapped['nii_6584_flux_err'].values,
    overlapped['oiii_5007_flux_err'].values,
    overlapped['D4000_ERR'].values,
    overlapped['cModelMagErr_r'].values,
    # np.sqrt(1 / overlapped['cModelFluxIvar_r'].values)  # gets squared later
]).T.astype(np.float32))

varnames = ['f150', 'z', 'logM', 'ha', 'hb', 'nii', 'oiii', 'D4000', 'R']

cov = torch.diag_embed(Xerr)**2.

assert torch.isfinite(cov).all().numpy()
assert torch.isfinite(Xerr).all().numpy()
assert torch.isfinite(Xdata).all().numpy()


# TODO test normalising data a little better
cap = 20.


def inv_logit(x):
    y = torch.exp(x)
    return y / (1 + y)

np.testing.assert_allclose(inv_logit(torch.logit(torch.scalar_tensor(0.3425))).numpy(), 0.3425)

TransformPair = namedtuple('TransformPair', ['data2fitting', 'fitting2data', 'jacobian', 'prefilter'])

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


def make_data_transform(data, function=None, inv_function=None,
                   function_jacobian=None, prefilter=None):
    """
    Make a normalising transform on function(x) given function(data)
    after removing from data those values which do not satisfy `prefilter` condition.
    data: the observed dataset, all of it
    function: is the function that maps from data to fitting space
    inv_function: is the function that maps from fitting space to data space
    function_jacobian: is the jacobian of the function that maps data->fitting
    prefilter: is the function that returns True/False to include data in the calculation of mean and std
                if the transform is log/exp then you should prefilter with x > 0
    """
    if prefilter is None:
        original = data
    else:
        original = data[prefilter(data)]
    if function is None:
        function = lambda x: x
        inv_function = lambda x: x
        function_jacobian = lambda x, y: torch.ones_like(x)
    elif function_jacobian is None or inv_function is None:
        raise ValueError(f"If you define `function` you must define its inverse and the inverse's jacobian")
    transformed = function(original)
    mu, std = transformed.mean(dim=0), transformed.std(dim=0)

    def forward(x):
        return (function(x) - mu) / std

    def backward(y):
        return inv_function((y * std) + mu)

    def jac(x, y):
        """
        df^{-1}(y) / dy
        where y is the fitting and x is the data
        """
        return torch.clamp(function_jacobian(x, y) / std, min=1e-30)

    return TransformPair(forward, backward, jac, prefilter)

def transform_from_flow_transform(data, Transform: nflows.transforms.Transform, prefilter, **kwargs):
    transform = Transform(**kwargs)
    return make_data_transform(data,  lambda x: transform.forward(x)[0],
                               lambda x: transform.inverse(x)[1], lambda x: transform.inverse(x)[1],
                               prefilter)

class OffsetSigmoid(Sigmoid):

    def __init__(self, temperature=1, eps=1e-6, offset=0.5, learn_temperature=False):
        super().__init__(temperature, eps, learn_temperature)
        self.offset = offset

    def forward(self, inputs, context=None):
        x, dx = super().forward(inputs, context)
        return x - self.offset, dx

    def inverse(self, inputs, context=None):
        return super().inverse(inputs+self.offset, context)



def make_uncertainty_transform(data, uncertainty):
    def function(x, xerr):
        return torch.log(torch.abs(xerr / x))

    def inv_function(x, ylogsnr):
        return torch.abs(torch.exp(ylogsnr) * x)

    transformed = function(data, uncertainty)
    mu, std = transformed.mean(dim=0), transformed.std(dim=0)
    def forward(x, xerr):
        return (function(x, xerr) - mu) / std

    def backward(x, ylogsnr):
        return inv_function(x, (std * ylogsnr) + mu)

    return TransformPair(forward, backward, None, None)


radio_flux_transform = make_data_transform(Xdata[:, 0], torch.asinh,
                                           lambda y: torch.sinh(torch.clamp(y, -cap, cap)),
                                           lambda x, y: torch.cosh(torch.clamp(y, -cap, cap)))
redshift_transform = make_data_transform(Xdata[:, 1])
mass_transform = make_data_transform(Xdata[:, 2])
line_transforms = [make_data_transform(Xdata[:, i],
                                       torch.asinh,
                                       lambda y: torch.sinh(torch.clamp(y, -cap, cap)),
                                       lambda x, y: torch.cosh(torch.clamp(y, -cap, cap)))
                   for i in range(3, 7)]
d4000_transform = make_data_transform(Xdata[:, 7])
rflux_transform = make_data_transform(Xdata[:, 8])

model_bounds = [
    [0, 500],
    [0.02, 0.09],
    [2, 15],
] + \
    [[-10_000, 10_000]]*4 \
+ [
    [0, 10],
    [0, 20],
]

data_transforms = [radio_flux_transform, redshift_transform, mass_transform] + line_transforms + [d4000_transform, rflux_transform]

selected_names = ['f150', 'logM', 'ha', 'hb', 'nii', 'oiii', 'D4000', 'R']
indexes = [varnames.index(i) for i in selected_names]

Xdata = Xdata[:, indexes]
Xerr = Xerr[:, indexes]
data_transforms = [data_transforms[i] for i in indexes]
error_transform = make_uncertainty_transform(Xdata, Xerr)


def data2fitting(x):
    """
    Transform the data matrix x into the fitting regime
    if generated_samples is True, x is assumed to have been generated from our model before
    """
    return torch.stack([t.data2fitting(xi) for xi, t in zip(x.T, data_transforms)]).T
# data2fitting.names = ['asinh(f150_scaled)', 'z', 'logM', 'asinh(ha)', 'asinh(hb)', 'asinh(nii)', 'asinh(oiii)',
#                       'D4000', 'cMag_r']
data2fitting.names = [f'T({i})' for i in selected_names]

def fitting2data(y):
    """
    Transform the fitting matrix x into the data regime
    if generated_samples is True, x is assumed to have been generated from our model before
    """
    return torch.stack([t.fitting2data(yi) for yi, t in zip(y.T, data_transforms)]).T
# fitting2data.names = ['f150_scaled', 'z', 'logM', 'ha', 'hb', 'nii', 'oiii', 'D4000', 'cMag_r']
fitting2data.names = selected_names


def jacobian(x, y):
    """
    Returns jacobian, is_diag={True/False}
    """
    return torch.stack([t.jacobian(xi, yi) for xi, yi, t in zip(x.T, y.T, data_transforms)]).T, True


def data2auxillary(x):
    """
    Adds other dimension that may be used for plotting/diagnostics etc
    Not used in the actual fit
    """
    return torch.tensor([[]], dtype=Xfitting.dtype).T
    x = x.T
    return torch.stack([
        torch.log10(x[3]) - torch.log10(x[5]),  # line_scale is not needed since it's a ratio
        torch.log10(x[4]) - torch.log10(x[6]),
        # 22.5 - (2.5 * torch.log10(x[-1])),
    ]).T
data2auxillary.names = []#['log(nii / Ha)', 'log(oiii / Hb)']#, 'cmag_r']


from chainconsumer import ChainConsumer


Xfitting = data2fitting(Xdata)  # will contain nans from fluxes
Xerrfitting = error_transform.data2fitting(Xdata, Xerr)
np.testing.assert_allclose(fitting2data(Xfitting), Xdata, rtol=0.07)
# np.testing.assert_allclose(error_transform.fitting2data(fitting2data(Xfitting), Xerrfitting), Xerr, rtol=0.008)

class DeconvGaussianTransformed(Distribution):
    """
    Same as the original but the uncertainty gaussians are uncorrelated.
    Here the uncertainty covariance is assumed to be diagonal, but this is not checked!
    You also specify a transform from fitting space to data space. In this case, the uncertainty
    is assumed to be defined in the data space, not the fitting space.
    """

    def __init__(self, use_diag_errs) -> None:
        super().__init__()
        self.use_diag_errs = use_diag_errs
        if 'f150' in fitting2data.names[0]:
            print('subtracting noise from radio flux')
        else:
            print('NOT doing radio flux noise subtraction')

    def convolve_noise(self, observed, observed_err_tril, intrinsic):
        """
        observed: contains actual flux data, can be negative or positive
        observed_err_tril: contains observed flux error tril matrix
        intrinsic: the "true" values generated by the model
        returns: P(data|intrinsic,data_err) == p(w|v)
        """
        if 'f150' in fitting2data.names:
            f150_index = fitting2data.names.index('f150')
            logps = []
            for mu, sigma, alpha in zip(random_xdgmm.mu, random_xdgmm.V, random_xdgmm.alpha):
                mu = torch.as_tensor(mu)
                sigma = torch.as_tensor(sigma)
                if self.use_diag_errs:
                    err = torch.sqrt(observed_err_tril[:, f150_index]**2. + sigma[0, 0])
                else:
                    raise NotImplementedError(f"Wrong")
                    err = torch.sqrt(observed_err_tril[:, f150_index, f150_index] + sigma[0, 0])
                flux_logp = Normal(loc=mu+intrinsic[:, f150_index], scale=err).log_prob(observed[:, f150_index])
                logps.append(flux_logp + torch.log(torch.as_tensor(alpha, dtype=err.dtype)))
            flux_logp = torch.logsumexp(torch.stack(logps), dim=0)
        else:
            flux_logp = 0.
        other_logps = []
        if self.use_diag_errs:
            for i in [j for j, n in enumerate(fitting2data.names) if 'f150' not in n]:
                other_logps.append(Normal(loc=intrinsic[:, i], scale=observed_err_tril[:, i]).log_prob(observed[:, i]))
            return flux_logp + sum(other_logps)
        return LocDifferentiableMultivariateNormal(loc=intrinsic, scale_tril=observed_err_tril).log_prob(observed)

    def log_prob(self, inputs, context):
        Z, noise_l = inputs  # fitting space
        X = fitting2data(Z)
        if self.use_diag_errs:
            noise_l = error_transform.fitting2data(X, noise_l)
        context_dataspace = fitting2data(context)
        jac, is_diag = jacobian(context_dataspace, context)  # jacobian of the data->fitting transform
        if is_diag:
            log_scaling = torch.log(torch.abs(jac)).sum(axis=1)
        else:
            log_scaling = torch.slogdet(jac)[1]
        context_dataspace = torch.where(torch.isfinite(context_dataspace), context_dataspace,
                                          torch.scalar_tensor(torch.finfo(context_dataspace.dtype).min,
                                                              dtype=context_dataspace.dtype))
        log_scaling = torch.where(torch.isfinite(log_scaling), log_scaling,
                                          torch.scalar_tensor(torch.finfo(log_scaling.dtype).min,
                                                              dtype=log_scaling.dtype))
        return self.convolve_noise(X, noise_l, context_dataspace) - log_scaling


class MySVIFlow(SVIFlow):
    def _create_likelihood(self, use_diag_errs=False):
        return DeconvGaussianTransformed(use_diag_errs)


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

    def configure(self, statistics="max", max_ticks=5, plot_hists=True, flip=True, serif=True, sigma2d=False, sigmas=None, summary=None, bins=None, cmap=None, colors=None, linestyles=None, linewidths=None, kde=False, smooth=None, cloud=None, shade=None, shade_alpha=None, shade_gradient=None,
                  bar_shade=None, num_cloud=None, color_params=None, plot_color_params=False, cmaps=None, plot_contour=None, plot_point=None, show_as_1d_prior=None, global_point=True, marker_style=None, marker_size=None, marker_alpha=None, usetex=True, diagonal_tick_labels=True, label_font_size=12,
                  tick_font_size=10, spacing=None, contour_labels=None, contour_label_font_size=10, legend_kwargs=None, legend_location=None, legend_artists=None, legend_color_text=True, watermark_text_kwargs=None, summary_area=0.6827, zorder=None, stack=False,
                  logged=False):
        super().configure(statistics, max_ticks, plot_hists, flip, serif, sigma2d, sigmas, summary, bins, cmap, colors, linestyles, linewidths, kde, smooth, cloud, shade, shade_alpha, shade_gradient, bar_shade, num_cloud, color_params, plot_color_params, cmaps, plot_contour, plot_point,
                                 show_as_1d_prior, global_point, marker_style, marker_size, marker_alpha, usetex, diagonal_tick_labels, label_font_size, tick_font_size, spacing, contour_labels, contour_label_font_size, legend_kwargs, legend_location, legend_artists, legend_color_text,
                                 watermark_text_kwargs, summary_area, zorder, stack)
        self.config["logged"] = logged



def plot_samples(chains, record_df=None, c=None,
                 parameters=None, log_scales=None, perc=(0.0001, 99.),
                 log_densities=False):
    if c is None:
        c = MyChainConsumer()
    else:
        for name in chains:
            c.remove_chain(name)
    for name, chain in chains.items():
        chains[name] = torch.concat([torch.as_tensor(i) for i in chain  if len(i) > 0], axis=1).numpy()

    # de-duplicate
    names = fitting2data.names + data2fitting.names + data2auxillary.names
    _n = []
    _names = []
    for i, name in enumerate(names):
        if name not in _names:
            _n.append(i)
            _names.append(name)
    names = _names
    for name, chain in chains.items():
        chains[name] = chain[:, _n]
    zipped_chains = zip(*[i.T for i in chains.values()])
    extents = np.array([[np.nanpercentile(i[np.isfinite(i)], perc) for i in z] for z in zipped_chains])
    extents = [(np.min(z), np.max(z)) for z in extents]
    if parameters is None:
        parameters = names
    invalid_names = [n for n in parameters if n not in names]
    if invalid_names:
        raise KeyError(f"Names {invalid_names} are not known, available parameters are {names}")
    for name, chain in chains.items():
        filt = reduce(lambda a, b: a & b, ((o > l) & (o < u) for o, (l, u) in zip(chain.T, extents)))
        c.add_chain(chain[filt],
                    parameters=names,
                    name=name,
                    kde=False)
    c.configure(usetex=False, smooth=0, cloud=True, shade_alpha=0.15, logged=log_densities)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        c.plotter.plot(parameters=[n for n in names if n in parameters], log_scales=log_scales)
    if not hasattr(c, 'axins_losses') and record_df is not None:
        # axes = [ax.get_position() for ax in c.plotter.fig_data[1][0] if not ax.get_frame_on()]
        if np.size(c.plotter.fig_data[1]) == 4:
            c.axins_losses = c.plotter.fig_data[0].add_axes([0.75, 0.75, 0.2, 0.2])
        else:
            c.axins_losses = c.plotter.fig_data[0].add_axes([0.5, 0.6, 0.4, 0.3])
        # elif np.size(c.plotter.fig_data[1]) == 2:
        #     c.axins_losses = c.plotter.fig_data[0].add_axes(axes[0])
        # else:
        #     axes = [ax.get_position() for ax in c.plotter.fig_data[1][0]]
        #     c.axins_losses = c.plotter.fig_data[0].add_axes(axes[0])
        #     mn = min([ax.x0 for ax in axes]), min([ax.y0 for ax in axes])
        #     mx = max([ax.x0+ax.width for ax in axes]), max([ax.y0+ax.height for ax in axes])
        #     pos = mn + (mx[0] - mn[0], mx[1] - mn[1])
        #     c.axins_losses = c.plotter.fig_data[0].add_axes(pos)
        c.axins_losses_twin = c.axins_losses.twinx()
    if record_df is not None:
        if len(record_df):
            c.axins_losses.clear()
            c.axins_losses_twin.clear()
            c.axins_losses.plot(range(len(record_df)), record_df['train'], 'b-', label='train')
            c.axins_losses_twin.plot(range(len(record_df)), record_df['val'], 'r-', label='val')
            c.axins_losses.set_xlim(max([0, max(record_df['epoch']) - 20]), max(record_df['epoch']))
        autoscale_y(c.axins_losses)
        try:
            autoscale_y(c.axins_losses_twin)
        except ValueError:
            pass
        # c.axins_losses.legend(loc='lower right')
        plt.suptitle(f'epoch {len(record_df)}')
    return c


def save_epoch_to_disk(record_df, i, _svi, train_loss, val_loss, logl, kl):
    torch.save(_svi.model.state_dict(), params_dir / f'{i}.pt')
    torch.save(_svi.scheduler.state_dict(), params_dir / f'{i}.opt')
    torch.save(_svi.optimiser.state_dict(), params_dir / f'{i}.spt')
    record_df = record_df.append(
        {'epoch': i, 'train': train_loss, 'val': val_loss, 'logl': logl, 'kl': kl},
        ignore_index=True
    )
    record_df.to_csv(directory / f'record.csv')
    return record_df

def load_epoch_from_disk(i, _svi):
    _svi.model.load_state_dict(torch.load(params_dir / f'{i}.pt'))
    _svi.scheduler.load_state_dict(torch.load(params_dir / f'{i}.spt'))
    _svi.optimiser.load_state_dict(torch.load(params_dir / f'{i}.opt'))
    df = pd.read_csv(directory / f'record.csv').iloc[:i]
    return df


S = cov
partition = 0
limit = 5_000
# TODO: errors should be normalised as well!
if partition == 0:
    train_data = DeconvDataset(Xfitting[:limit], Xerrfitting[:limit], diag=True)
    test_data = None
else:
    split_n = min([Xfitting.shape[0], limit]) // partition
    X_test = Xfitting[:split_n]
    S_test = Xerrfitting[:split_n]
    X_train = Xfitting[split_n:][:limit-split_n]
    S_train = Xerrfitting[split_n:][:limit-split_n]
    print(f"validation={len(X_test)}, training={len(X_train)} ({1 / partition:.2%})")
    train_data = DeconvDataset(X_train, S_train, diag=True)
    test_data = DeconvDataset(X_test, S_test, diag=True)

# directory = Path('models/noisy-all-uncut-all-expect-f150')
directory = Path('models/noisy-all-uncut-all-expect-f150-without-redshift')
space_dir = directory / 'plots'
params_dir = directory / 'params'
directory.mkdir(parents=True, exist_ok=True)
space_dir.mkdir(parents=True, exist_ok=True)
params_dir.mkdir(parents=True, exist_ok=True)
print(f'saving to {directory}')

total_epochs = 1000
start_from = 0
svi = MySVIFlow(
    Xdata.shape[1],
    flow_steps=5,
    device= torch.device('cpu'),
    batch_size=1000,
    epochs=total_epochs,
    lr=1e-3,
    warmup=10,
    n_samples=25,
    grad_clip_norm=100,
    # bounds=data2fitting(torch.as_tensor(np.asarray(model_bounds).T)).T.numpy(),
    kl_multiplier=1.,
    scheduler_kwargs={'patience': 20},  # add overrides here
    use_iwae=True,
    use_diag_errs=True,
)
prior_samples = svi.model._prior.sample(10_000)
post_samples = svi.model._approximate_posterior.sample(10_000)
if start_from > 0:
    df = load_epoch_from_disk(start_from, svi)
else:
    df = pd.DataFrame({'train': [], 'val': [], 'logl': [], 'kl': []})

def iterator():
    for i in enumerate(svi.iter_fit(train_data, test_data, seed=SEED, num_workers=8,
                                    rewind_on_inf=True, return_kl_logl=True,
                                    start_from=start_from), start_from):  # iterator
        yield i


iterations = iterator()

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


def plot_epoch(i, _svi, xfitting, record_df=None, chainconsumer=None, parameters=None,
               log_scales=None, perc=(0.0001, 99.), log_densities=False):
    yfitting = _svi.sample_prior(len(xfitting))[0]
    ydata = fitting2data(yfitting)
    if chainconsumer is not None:
        for ax in chainconsumer.plotter.fig_data[0].axes:
            ax.clear()
    xdata = fitting2data(xfitting)
    xaux = data2auxillary(xdata)
    yaux = data2auxillary(ydata)
    chainconsumer = plot_samples(dict(observed=(xdata, xfitting, xaux), fit=(ydata, yfitting, yaux)), record_df, chainconsumer,
                                 parameters, log_scales, perc, log_densities)
    fig = chainconsumer.plotter.fig_data[0]
    plt.suptitle(f'epoch {i}')
    return chainconsumer

def plot_epoch_q(i, _svi, data_test_points, data_test_point_errs, record_df=None, chainconsumer=None, parameters=None, log_scales=None,
                 perc=(0.0001, 99.), log_densities=False, n=1_000_000):
    data_test_points = torch.as_tensor(data_test_points).to(svi.device)
    data_test_point_errs = torch.as_tensor(data_test_point_errs).to(svi.device)
    fitting_test_points = data2fitting(data_test_points)
    fitting_test_point_errs = error_transform.data2fitting(data_test_points, data_test_point_errs)
    Yfitting = _svi.sample_prior(n)[0]
    Qfitting = utils.merge_leading_dims(_svi.sample_posterior([fitting_test_points, fitting_test_point_errs], n), num_dims=2)
    Pfitting = utils.merge_leading_dims(_svi.resample_posterior([fitting_test_points, fitting_test_point_errs], n), num_dims=2)
    Ydata = fitting2data(Yfitting)
    Yaux = data2auxillary(Ydata)
    Qdata = fitting2data(Qfitting)
    Qaux = data2auxillary(Qdata)
    Pdata = fitting2data(Pfitting)
    Paux = data2auxillary(Pdata)

    try:
        Udata = np.concatenate([np.random.multivariate_normal(m, l @ l.T, size=n//len(data_test_points))
                                                for m, l in zip(data_test_points, data_test_point_errs)])
    except ValueError:
        Udata = np.random.normal(data_test_points, data_test_point_errs, size=(n, )+data_test_points.shape).reshape(-1, data_test_points.shape[-1])
    Udata = torch.as_tensor(Udata, dtype=Yfitting.dtype)
    Ufitting = data2fitting(Udata)
    Uaux = data2auxillary(Udata)
    if chainconsumer is not None:
        for ax in chainconsumer.plotter.fig_data[0].axes:
            ax.clear()
    chainconsumer = plot_samples(dict(fit=(Ydata, Yfitting, Yaux),
                                      uncertainty=(Udata, Ufitting, Uaux),
                                      q=(Qdata, Qfitting, Qaux),
                                      p=(Pdata, Pfitting, Paux),
                                      ),
                                 None, chainconsumer,
                                 parameters, log_scales, perc, log_densities)
    fig = chainconsumer.plotter.fig_data[0]
    plt.suptitle(f'epoch {i}')
    return chainconsumer

if __name__ == '__main__':
    for i, o in iterations:
        df = save_epoch_to_disk(df, i, *o)