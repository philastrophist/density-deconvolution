import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fit_data_flux_mass import *

i = 9990
svi.model.load_state_dict(torch.load(params_dir / f'{i}.pt'))
try:
    df = pd.read_csv(params_dir / f'record.csv').iloc[:start_from]
except IOError:
    df = None
# fig = plot_epoch(i, svi, df)
# fig.savefig(space_dir / f'{i}.png')
# Xaux = data2auxillary(Xdata)
# Yfitting = svi.sample_prior(len(Xdata))
# Ydata = fitting2data(Yfitting)[0]
# Yaux = data2auxillary(Ydata)
from deconv.gmm.plotting import plot_covariance

mu = np.array([  # in fitting space
    [2.4, 2.5, -2.1, 7, 7, 0., 0.]
])
err = np.array(  # in data space
    [0.05, 0.001, 0.01, 0.01, 0.01, 0.01, 0.01]
)

V = np.diag(err**2.)[None, ...]
test_point = [
    torch.Tensor(mu).to(svi.device),
    vmap(torch.cholesky)(torch.Tensor(V)).to(svi.device)
]
Udata = np.concatenate([np.random.multivariate_normal(m, v, size=10_000) for m, v in zip(fitting2data(test_point[0]), V)])
Ufitting = data2fitting(torch.as_tensor(Udata))
Qfitting = svi.resample_posterior(test_point, 10_000)[0]
Qdata = fitting2data(Qfitting.reshape(-1, Qfitting.shape[-1])).reshape(Qfitting.shape)

Yfitting = svi.sample_prior(2_000_000)[0]
Ydata = fitting2data(Yfitting)

Yaux = data2auxillary(Ydata).numpy()
Uaux = data2auxillary(torch.as_tensor(Udata)).numpy()
Qaux = data2auxillary(torch.as_tensor(Qdata)).numpy()
Xaux = data2auxillary(torch.as_tensor(Xdata)).numpy()

names = fitting2data.names + data2fitting.names + data2auxillary.names
X = np.concatenate([Xdata, Xfitting, Xaux], axis=1)
Q = np.concatenate([Qdata, Qfitting, Qaux], axis=1)
U = np.concatenate([Udata, Ufitting, Uaux], axis=1)
Y = np.concatenate([Ydata, Yfitting, Yaux], axis=1)

dimensions = names#['logM', 'log(L150)']
dims = [names.index(i) for i in dimensions]


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
plot_samples(Xdata, Xfitting, Xaux, Ydata, Yfitting, Yaux)

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
