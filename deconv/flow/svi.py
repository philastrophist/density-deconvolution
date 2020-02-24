import torch
import torch.utils.data as data_utils

from nsflow.nde import flows, transforms
from nsflow.nde.distributions import ConditionalDiagonalNormal, StandardNormal
from nsflow.vae import VariationalAutoencoder

from .distributions import DeconvGaussian
from .maf import MAFlow
from .nn import DeconvInputEncoder

class SVIFLow(MAFlow):

    def __init__(self, dimensions, flow_steps, lr, epochs, context_size=64,
                 batch_size=256, kl_warmup=0.1, kl_init_factor=0.5,
                 device=None):
        super().__init__(
            dimensions, flow_steps, lr, epochs, batch_size, device
        )
        self.context_size = context_size
        self.kl_warmup = kl_warmup
        self.kl_init_factor = kl_init_factor

        self.model = VariationalAutoencoder(
            prior=self._create_prior(),
            approximate_posterior=self._create_approximate_posterior(),
            likelihood=self._create_likelihood(),
            inputs_encoder=self._create_input_encoder()
        )

        self.model.to(self.device)

    def _create_prior(self):
        self.transform = self._create_transform()
        distribution = StandardNormal((self.dimensions,))
        return flows.Flow(
            self.transform,
            distribution
        )

    def _create_likelihood(self):
        return DeconvGaussian()

    def _create_input_encoder(self):
        return DeconvInputEncoder(self.dimensions, self.context_size)

    def _create_approximate_posterior(self):

        # context_encoder = torch.nn.Linear(
        #     self.context_size,
        #     2 * self.dimensions
        # )

        # distribution = ConditionalDiagonalNormal(
        #     shape=(self.dimensions,),
        #     context_encoder=context_encoder
        # )

        distribution = StandardNormal((self.dimensions,))

        posterior_transform = self._create_transform(self.context_size)

        return flows.Flow(
            transforms.InverseTransform(
                posterior_transform
            ),
            distribution
        )

    def _kl_factor(self, step, max_steps):

        f = min(
            1.0,
            self.kl_init_factor + (
                (1 - self.kl_init_factor) * step / (
                    self.kl_warmup * max_steps
                )
            )
        )
        return f


    def fit(self, data, val_data=None):

        optimiser = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.lr
        )

        loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True
        )

        batches = len(loader)

        max_steps = self.epochs * batches

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            max_steps
        )

        for i in range(self.epochs):

            self.model.train()

            train_loss = 0.0

            for j, d in enumerate(loader):

                d = [a.to(self.device) for a in d]

                optimiser.zero_grad()

                step = i * batches + j

                elbo = self.model.stochastic_elbo(
                    d,
                    num_samples=1,
                    kl_multiplier=self._kl_factor(step, max_steps)
                )

                train_loss += torch.sum(elbo).item()
                loss = -1 * torch.mean(elbo)
                loss.backward()
                optimiser.step()

                scheduler.step(step)

            train_loss /= len(data)

            if val_data:
                val_loss = self.score_batch(val_data) / len(val_data)
                print('Epoch {}, Train Loss: {}, Val Loss: {}'.format(
                    i,
                    train_loss,
                    val_loss
                ))
            else:
                print('Epoch {}, Train Loss: {}'.format(i, train_loss))


    def score(self, data, log_prob=False):
        with torch.no_grad():
            if log_prob:
                return self.model.log_prob_lower_bound(data)
            else:
                return self.model.stochastic_elbo(data)

    def score_batch(self, dataset, log_prob=False):
        loader = data_utils.DataLoader(
            dataset,
            batch_size=self.batch_size
        )
        score = 0.0

        for j, d in enumerate(loader):
            d = [a.to(self.device) for a in d]
            score += torch.sum(self.score(d, log_prob)).item()

        return score