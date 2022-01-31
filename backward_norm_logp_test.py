import torch
from torch.distributions import MultivariateNormal

torch.autograd.set_detect_anomaly(True)

Z = torch.tensor([[1e+2000, 1e+2000],
                  [1e+2000, 1e+2000]], requires_grad=True, dtype=float)

# need to catch nans before they are created
Z = torch.where(Z > torch.scalar_tensor(1e+20, dtype=float), torch.scalar_tensor(1., dtype=float), Z)
n = -Z**Z
# n = torch.where(torch.isfinite(n), n , torch.finfo().max)
logl = torch.logsumexp(n, dim=1)
# logl = MultivariateNormal(loc=torch.zeros((1, 2), dtype=float),
#                           scale_tril=torch.cholesky(torch.eye(2, dtype=float)/10000)).log_prob(Z)
# print(logl)
logl.sum().backward()