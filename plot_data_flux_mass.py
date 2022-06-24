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

    def __init__(self, names=None):
        if names is None:
            names = varnames
        self.names = names

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
        self.c = plot_epoch(self.i, self.svi, train_data.X, self.df, self.c, self.names)

    def plot_epoch_q(self, test_point, test_point_err, i=None, force=False):
        updated = self.update(i)
        if not updated and self.qc is not None and not force:
            return self.i, self.qc.plotter.fig_data[0]
        self._reset_c_axes(self.qc)
        self.qc = plot_epoch_q(self.i, self.svi, test_point, test_point_err, self.df, self.qc, self.names, n=10_000, perc=(0., 100.))


if __name__ == '__main__':

    means = torch.median(fitting2data(train_data.X), axis=0, keepdims=True)[0]
    errs = torch.median(error_transform.fitting2data(fitting2data(train_data.X), train_data.noise_covars), axis=0, keepdims=True)[0]

    plotter = Plotter(['T(R)', 'T(M)'])
    i = get_newest()
    print(i)
    plotter.plot_epoch(i)
    # plotter.plot_epoch_q(means, errs, i)

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
                print(i)
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

    plt.show()
