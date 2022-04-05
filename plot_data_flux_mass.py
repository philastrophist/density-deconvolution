import sys
import time
from copy import copy

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.special import logsumexp
from xdgmm import XDGMM as XDGMM_

import matplotlib as mpl
mpl.rcParams['keymap.back'].remove('left')
mpl.rcParams['keymap.forward'].remove('right')


from fit_data_flux_mass_noisy import *

def get_newest():
    try:
        return int(max(params_dir.glob('*.pt'), key=lambda p: p.stat().st_ctime).name[:-3])
    except ValueError as e:
        raise FileNotFoundError(f"No parameter files can be found in `{params_dir}`") from e

# class XDGMM(XDGMM_):
#     def fit(self, X, Xerr, assignments=None):
#         """Fit the XD model to data
#
#         Whichever method is specified in self.method will be used.
#
#         Results are saved in self.mu/V/weights and in the self.GMM
#             object
#
#         Parameters
#         ----------
#         X: array_like, shape = (n_samples, n_features)
#             Input data.
#         Xerr: array_like, shape = (n_samples, n_features, n_features)
#             Error on input data.
#         """
#
#         if type(X) == pd.core.frame.DataFrame:
#             if type(X.columns) == pd.indexes.base.Index:
#                 self.labels = np.array(X.columns)
#             X = X.values
#
#         if self.method == 'astroML':
#             if assignments is not None:
#                 raise ValueError(f"Only method='Bovy' can handle assignments")
#             self.GMM.n_components = self.n_components
#             self.GMM.n_iter = self.n_iter
#             self.GMM.fit(X, Xerr)
#
#             self.V = self.GMM.V
#             self.mu = self.GMM.mu
#             self.weights = self.GMM.alpha
#
#         if self.method == 'Bovy':
#             """
#             Bovy extreme_deconvolution only imports if the method is
#                 'Bovy' (this is because installation is somewhat more
#                 complicated than astroML, and we don't want it to be
#                 required)
#
#             As with the astroML method, initialize with a few steps of
#                 the scikit-learn GMM
#             """
#             from extreme_deconvolution import extreme_deconvolution \
#                 as bovyXD
#             from sklearn.mixture import GaussianMixture as skl_GMM
#
#             tmp_gmm = skl_GMM(self.n_components, max_iter=10,
#                               covariance_type='full',
#                               random_state=self.random_state)
#             tmp_gmm.fit(X)
#             self.mu = tmp_gmm.means_
#             self.weights = tmp_gmm.weights_
#             self.V = tmp_gmm.covariances_
#
#             logl = bovyXD(X, Xerr, self.weights, self.mu, self.V,
#                           tol=self.tol, maxiter=self.n_iter, w=self.w)
#             self.GMM.V = self.V
#             self.GMM.mu = self.mu
#             self.GMM.alpha = self.weights
#
#         return self
#
#     def convolution(self, X, Xerr, n):
#         samples = []
#         logweights = []
#         for m, v, alpha in zip(self.mu, self.V, self.weights):
#             logcc = np.log(alpha) + stats.multivariate_normal(m, v + Xerr, True).logpdf(X)
#             invv = np.linalg.inv(v)
#             invxerr = np.linalg.inv(Xerr)
#             inv = np.linalg.inv(invv + invxerr)
#             mc = inv @ ((invv @ m) + (invxerr @ X))
#             samples.append(stats.multivariate_normal(mc, inv).rvs(n))
#             logweights.append([logcc]*n)
#         return np.array(samples).ravel(), np.array(logweights).ravel(), logsumexp(logcc, axis=0)

# xdgmm = XDGMM(1, method='Bovy')
# # works just for this example logM
# xdgmm.fit(fitting2data(train_data.X).numpy(),
#           vmap(lambda x: x @ x.T)(train_data.noise_covars).numpy())

class Plotter:
    previous = -2
    qc = None
    c = None
    i = None

    def update(self, i=None):
        if i is None:
            self.i = get_newest()
            i = self.i
        try:
            self.df = pd.read_csv(directory / f'record.csv').iloc[:i+1]
        except IOError:
            self.df = None
        if self.previous == i:
            return False
        self.previous = i
        while True:  # wait for file
            try:
                svi.model.load_state_dict(torch.load(params_dir / f'{i}.pt'))
                break
            except RuntimeError:
                time.sleep(0.5)
        self.svi = svi
        return True

    def _reset_c_axes(self, c):
        if c is not None:
            for ax in c.plotter.fig_data[0].axes:
                ax.clear()

    def plot_epoch(self, i=None, force=False):
        try:
            updated = self.update(i)
        except EOFError:
            time.sleep(1.)
            updated = self.update(i)
        if not updated and self.c is not None and not force:
            return self.i, self.c.plotter.fig_data[0]
        self._reset_c_axes(self.c)
        self.c = plot_epoch(self.i, self.svi, train_data.X, self.df, self.c)

    def plot_epoch_q(self, test_point, test_point_err, i=None, force=False):
        updated = self.update(i)
        if not updated and self.qc is not None and not force:
            return self.i, self.qc.plotter.fig_data[0]
        self._reset_c_axes(self.qc)
        self.qc = plot_epoch_q(self.i, self.svi, test_point, test_point_err, self.df, self.qc, n=10_000, perc=(0., 100.))


means = torch.median(Xdata, axis=0, keepdims=True)[0]
errs = torch.median(Xerr, axis=0, keepdims=True)[0]

plotter = Plotter()
i = get_newest()
print(i)
plotter.plot_epoch(i)
# plotter.plot_epoch_q(means, errs, i)
# plotter.c.add_chain(random_xdgmm.sample(10_000), data2fitting.names, 'xdgmm')
# plotter.plot_epoch(i, force=True)
# plotter.plot_epoch_q(test_points, i)
# samples, logweights, logps = xdgmm.convolution(fitting2data(means).numpy(), covars.numpy()[0], 10_000)
# plotter.qc.add_chain(samples, data2fitting.names, 'xdgmm-posterior', weights=np.exp(logweights - logsumexp(logweights)))
# plotter.plot_epoch_q(test_points, i, force=True)

# us = np.random.uniform(8, 12.5, 1000)
# logps_estim = xdgmm.score(us[:, None], [np.eye(1)*1e-10]*len(us))
# score = plotter.svi.score(test_points, log_prob=True, num_samples=10_000).item()
# print(f"xdgmm logp = {logps:.4f}")
# print(f"estim. xdgmm logp = {logps_estim:.4f}")
# print(f"q lnscore = {score:.4f}")

# xdgmm_samples = xdgmm.sample(10_000)
fig = plotter.c.plotter.fig_data[0]
# fig = plotter.qc.plotter.fig_data[0]

keylist = []

def on_key_press(event):
    global i
    global keylist
    old_i = copy(i)
    if event.key == "left":
        keylist.append('left')
    elif event.key == "right":
        keylist.append('right')
    else:
        try:
            i = keylist.append(int(event.key))
        except ValueError:
            pass
    if event.key == 'enter' or event.key == 'left' or event.key == 'right':
        i = int(''.join(['0']+[str(k) for k in keylist if isinstance(k, int)]))
        for k in keylist:
            if k == 'left':
                i -= 1
            elif k == 'right':
                i += 1
        try:
            plotter.plot_epoch(i)
            fig.canvas.draw()
        except IOError:
            i = old_i
        del keylist[:]



# plt.gcf().canvas.mpl_connect("key_press_event", on_key_press)
anim1 = animation.FuncAnimation(fig, lambda i: plotter.plot_epoch(),
                                interval=500, repeat=False)

# anim2 = animation.FuncAnimation(plotter.qc.plotter.fig_data[0], lambda i: plotter.plot_epoch_q(test_points),
#                                 interval=500, repeat=False)

# fig.savefig(space_dir / f'{i}.png')
# Xaux = data2auxillary(Xdata)
# Yfitting = svi.sample_prior(len(Xdata))
# Ydata = fitting2data(Yfitting)[0]
# Yaux = data2auxillary(Ydata)
from deconv.gmm.plotting import plot_covariance
#
# mu = np.array([  # in fitting space
#     [2.4, 2.5, -2.1, 7, 7, 0., 0.]
# ])
# err = np.array(  # in data space
#     [0.05, 0.001, 0.01, 0.01, 0.01, 0.01, 0.01]
# )
#
# V = np.diag(err**2.)[None, ...]
# test_point = [
#     torch.Tensor(mu).to(svi.device),
#     vmap(torch.cholesky)(torch.Tensor(V)).to(svi.device)
# ]
# Udata = np.concatenate([np.random.multivariate_normal(m, v, size=10_000) for m, v in zip(fitting2data(test_point[0]), V)])
# Ufitting = data2fitting(torch.as_tensor(Udata))
# Qfitting = svi.resample_posterior(test_point, 10_000)[0]
# Qdata = fitting2data(Qfitting.reshape(-1, Qfitting.shape[-1])).reshape(Qfitting.shape)
#
# Yfitting = svi.sample_prior(2_000_000)[0]
# Ydata = fitting2data(Yfitting)
#
# Yaux = data2auxillary(Ydata).numpy()
# Uaux = data2auxillary(torch.as_tensor(Udata)).numpy()
# Qaux = data2auxillary(torch.as_tensor(Qdata)).numpy()
# Xaux = data2auxillary(torch.as_tensor(Xdata)).numpy()
#
# names = fitting2data.names + data2fitting.names + data2auxillary.names
# X = np.concatenate([Xdata, Xfitting, Xaux], axis=1)
# Q = np.concatenate([Qdata, Qfitting, Qaux], axis=1)
# U = np.concatenate([Udata, Ufitting, Uaux], axis=1)
# Y = np.concatenate([Ydata, Yfitting, Yaux], axis=1)
#
# dimensions = names#['logM', 'log(L150)']
# dims = [names.index(i) for i in dimensions]


# plt.scatter(*X[:, dims].T, s=1, alpha=0.5)
# plt.scatter(*U[:, dims].T, s=1, alpha=0.5)
# plt.scatter(*Q[:, dims].T, s=1, alpha=0.5)
# plt.xlabel(dimensions[0])
# plt.ylabel(dimensions[1])
from astroML.datasets.tools.sdss_fits import log_OIII_Hb_NII

# Y = Y[Y[:, names.index('ha')] < 0.] #[~np.isfinite(Y[:, names.index('log(nii / Ha)')]) & ~np.isfinite(Y[:, names.index('log(oiii / Hb)')])]
# plt.scatter(*X[:, dims].T, s=1, alpha=0.4)
# plt.scatter(*Y[:, dims].T, s=1, alpha=0.4)
# plt.xlim(8.5, 12)
# plt.ylim(16, 22)

# density plot
# plot_samples(Xdata, Xfitting, Xaux, Ydata, Yfitting, Yaux)

# logp plot of P(trueX | dataX, model)
# svi.model._approximate_posterior.log_prob()

# # logL, logM colour coded by fraction of AGN/SFG
# from astroML.datasets.tools.sdss_fits import log_OIII_Hb_NII
#
# Y = np.where(np.isfinite(Y), Y, np.nan)  # make sure all invalid values are nan for classification
# agn = Y[:, names.index('log(oiii / Hb)')] > log_OIII_Hb_NII(Y[:, names.index('log(nii / Ha)')])
# sfg = Y[:, names.index('log(oiii / Hb)')] < log_OIII_Hb_NII(Y[:, names.index('log(nii / Ha)')])
# neither = ~agn & ~sfg
#
#
# fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
#
# c = axs[0].hexbin(Y[:, names.index('logM')], Y[:, names.index('log(L150)')], C=sfg,
#            gridsize=50, mincnt=10, cmap='viridis',
#            reduce_C_function=np.mean, extent=[9, 12, 16, 21])
# divider = make_axes_locatable(axs[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(c, cax=cax, orientation='vertical').set_label('Fraction of SFG-unidentified')
#
#
# c = axs[1].hexbin(Y[:, names.index('logM')], Y[:, names.index('log(L150)')], C=agn,
#            gridsize=50, mincnt=10, cmap='inferno',
#            reduce_C_function=np.mean, extent=[9, 12, 16, 21])
# divider = make_axes_locatable(axs[1])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(c, cax=cax, orientation='vertical').set_label('Fraction of BPT-AGN')
#
# c = axs[2].hexbin(Y[:, names.index('logM')], Y[:, names.index('log(L150)')], C=neither,
#            gridsize=50, mincnt=10, cmap='magma',
#            reduce_C_function=np.mean, extent=[9, 12, 16, 21], vmin=0.0088, vmax=0.4)
# divider = make_axes_locatable(axs[2])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(c, cax=cax, orientation='vertical').set_label('Fraction of BPT-unidentified')
#
# axs[2].set_xlabel('logM')
# axs[1].set_ylabel('logL')
#
#
#
# fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
#
# c = axs[0].hexbin(Y[:, names.index('logM')], np.log10(Y[:, names.index('ha')]), C=sfg,
#            gridsize=100, mincnt=50, cmap='viridis',
#            reduce_C_function=np.mean, extent=[8, 12, -5, 5])
# divider = make_axes_locatable(axs[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(c, cax=cax, orientation='vertical').set_label('Fraction of SFG-unidentified')
#
#
# c = axs[1].hexbin(Y[:, names.index('logM')], np.log10(Y[:, names.index('ha')]), C=agn,
#            gridsize=100, mincnt=50, cmap='inferno',
#            reduce_C_function=np.mean, extent=[8, 12, -5, 5])
# divider = make_axes_locatable(axs[1])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(c, cax=cax, orientation='vertical').set_label('Fraction of BPT-AGN')
#
# c = axs[2].hexbin(Y[:, names.index('logM')], np.log10(Y[:, names.index('ha')]), C=neither,
#            gridsize=100, mincnt=50, cmap='magma',
#            reduce_C_function=np.mean, extent=[8, 12, -5, 5])
# divider = make_axes_locatable(axs[2])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(c, cax=cax, orientation='vertical').set_label('Fraction of BPT-unidentified')
#
# axs[2].set_ylabel('logha')
# axs[1].set_xlabel('logM')
#
#
# fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
# c = axs[0].hexbin(Y[sfg, names.index('logM')], Y[sfg, names.index('log(L150)')], C=Y[sfg, names.index('ha')],
#            gridsize=100, mincnt=50, cmap='magma',
#            reduce_C_function=np.mean,  extent=[9, 12, 16, 21], )#vmin=-2.5, vmax=0.)
# divider = make_axes_locatable(axs[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(c, cax=cax, orientation='vertical').set_label('Ha')
#
# c = axs[1].hexbin(Y[agn, names.index('logM')], Y[agn, names.index('log(L150)')], C=Y[agn, names.index('ha')],
#            gridsize=100, mincnt=50, cmap='magma',
#            reduce_C_function=np.mean,  extent=[9, 12, 16, 21], )#vmin=-2.5, vmax=0.)
# divider = make_axes_locatable(axs[1])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(c, cax=cax, orientation='vertical').set_label('Ha')
#
# c = axs[2].hexbin(Y[neither, names.index('logM')], Y[neither, names.index('log(L150)')], C=Y[neither, names.index('ha')],
#            gridsize=100, mincnt=50, cmap='magma',
#            reduce_C_function=np.mean,  extent=[9, 12, 16, 21], )#vmin=-2.5, vmax=0.)
# divider = make_axes_locatable(axs[2])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(c, cax=cax, orientation='vertical').set_label('Ha')

# pd.DataFrame(Y, columns=names).to_csv('svi_model_.csv')

# for ax in axs:
#     ax.scatter(X[:, names.index('log(L150)')], np.log10(X[:, names.index('ha')]), s=1, alpha=0.01)

# plt.figure()
# plt.scatter(*Y[sfg][:, [names.index('logM'), names.index('log(L150)')]].T, s=1, alpha=0.001, label='SFG')
# plt.scatter(*Y[neither][:, [names.index('logM'), names.index('log(L150)')]].T, s=1, alpha=0.01, label='no BPT')
# plt.scatter(*Y[agn][:, [names.index('logM'), names.index('log(L150)')]].T, s=1, alpha=0.01, label='AGN')
# plt.xlim(9, 12)
# plt.ylim(16, 21)

plt.show()
