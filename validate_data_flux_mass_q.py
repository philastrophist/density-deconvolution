import torch
from torch.special import ndtri

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

mu = np.array([  # in fitting space
    [2.4, 2.5, -0., 7, 7, 0., 0.]
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

Qfitting = svi.sample_posterior(test_point, 10_000)[0]
Qdata = fitting2data(Qfitting.reshape(-1, Qfitting.shape[-1])).reshape(Qfitting.shape)

Pfitting = svi.resample_posterior(test_point, 10_000)[0]
Pdata = fitting2data(Qfitting.reshape(-1, Qfitting.shape[-1])).reshape(Qfitting.shape)

Yfitting = svi.sample_prior(2_0000)[0]
Ydata = fitting2data(Yfitting)

Yaux = data2auxillary(Ydata).numpy()
Uaux = data2auxillary(torch.as_tensor(Udata)).numpy()
Qaux = data2auxillary(torch.as_tensor(Qdata)).numpy()
Xaux = data2auxillary(torch.as_tensor(Xdata)).numpy()
Paux = data2auxillary(torch.as_tensor(Pdata)).numpy()

names = fitting2data.names + data2fitting.names + data2auxillary.names
X = np.concatenate([Xdata, Xfitting, Xaux], axis=1)
Q = np.concatenate([Qdata, Qfitting, Qaux], axis=1)
P = np.concatenate([Pdata, Pfitting, Paux], axis=1)
U = np.concatenate([Udata, Ufitting, Uaux], axis=1)
Y = np.concatenate([Ydata, Yfitting, Yaux], axis=1)


dimensions = ['logM', 'log(L150)']
dims = [names.index(i) for i in dimensions]


from dynesty import NestedSampler
from dynesty import utils as dynfunc

class Pool:
    def map(self, func, x_array):
        with torch.no_grad():
            svi.model.eval()
            x = torch.as_tensor(x_array, dtype=Yfitting.dtype)
            return func(x).numpy()

def integrate_nested(logfunc, ndims, nlive, dlogz=None, show_bar=True):
    def prior_transform(u):
        x = ndtri(torch.as_tensor(u, dtype=Yfitting.dtype)[None])
        y, logabsdet = svi.model._prior._transform.inverse(x)
        return y[0].numpy()

    sampler = NestedSampler(logfunc, prior_transform, ndims, nlive=nlive)
    sampler.run_nested(dlogz=dlogz, print_progress=show_bar)
    r = sampler.results.logz[-1], sampler.results.logzerr[-1]
    return r[0], r[1], sampler.results


def lnfunc(x):
    """
    log posterior in latent space.
    x is in the latent space of the model prior i.e. it is drawn from the standard normal
    therefore, this function is just the likelihood of those points transformed after moving to the gaussian space
    """
    return svi.model._likelihood.log_prob(test_point, x)

integral, err, results = integrate_nested(lambda x: lnfunc(torch.as_tensor(x[None], dtype=Yfitting.dtype))[0].numpy(), len(Yfitting[0]), 100)
samples_data = fitting2data(torch.as_tensor(results.samples))
samples_aux = data2auxillary(samples_data)
_samples = np.concatenate([samples_data.numpy(), results.samples, samples_aux.numpy()], axis=1)
weights = np.exp(results.logwt - dynfunc.logsumexp(results.logwt))
samples = dynfunc.resample_equal(_samples, weights)

plt.scatter(*Y[:, dims].T, s=1, alpha=0.3, label='model')
plt.scatter(*U[:, dims].T, s=1, alpha=0.3, label='uncertainty(v | w)')
plt.scatter(*Q[:, dims].T, s=1, alpha=0.3, label='q(v)')
plt.scatter(*P[:, dims].T, s=1, alpha=0.3, label='p(w) ~ P(v|w)P(v) / q(v)')
plt.scatter(*samples[:, dims].T, s=1, alpha=0.3, label='dynesty p(w)')
plt.legend( markerscale=3)

svi_lnprob = svi.score(test_point, True, 10_000).numpy()[0]
print(f"dynesty lnp(w) = {integral:.2f}")
print(f"flow q lnp(w) = {svi_lnprob:.2f}")

plt.show()

