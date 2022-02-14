from copy import deepcopy

import numpy as np
import torch
import torch.utils.data as data_utils
from torch.nn.utils import clip_grad_norm_

from nflows import flows, transforms, utils
from nflows.distributions import ConditionalDiagonalNormal, StandardNormal
from tqdm import tqdm

from .distributions import DeconvGaussian
from .maf import MAFlow
from .nn import DeconvInputEncoder
from .vae import VariationalAutoencoder
from ..utils.sampling import minibatch_sample


def seed_worker(worker_id):
    import numpy as np
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class HandleInfFlow(flows.Flow):
    def _log_prob(self, inputs, context):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        noise = torch.where(torch.isfinite(noise), noise, torch.scalar_tensor(-1e+10, dtype=noise.dtype))
        log_prob = self._distribution.log_prob(noise, context=embedded_context) + logabsdet
        return torch.where(torch.isfinite(log_prob), log_prob, torch.scalar_tensor(-1e+10, dtype=log_prob.dtype))


class SVIFlow(MAFlow):

    def __init__(self, dimensions, flow_steps, lr, epochs, context_size=64, hidden_features=128,
                 batch_size=256, kl_warmup=0.2, kl_init_factor=0.5,
                 n_samples=50, grad_clip_norm=None, use_iwae=False, device=None):
        super().__init__(
            dimensions, flow_steps, lr, epochs, batch_size, device
        )
        self.context_size = context_size
        self.hidden_features = hidden_features
        self.kl_warmup = kl_warmup
        self.kl_init_factor = kl_init_factor
        
        self.n_samples = n_samples
        self.grad_clip_norm = grad_clip_norm
        self.use_iwae = use_iwae

        self.model = VariationalAutoencoder(
            prior=self._create_prior(),
            approximate_posterior=self._create_approximate_posterior(),
            likelihood=self._create_likelihood(),
            inputs_encoder=self._create_input_encoder()
        )

        self.model.to(self.device)

    def _create_prior(self):
        self.transform = self._create_transform(context_features=None, hidden_features=self.hidden_features)
        distribution = StandardNormal((self.dimensions,))
        return HandleInfFlow(
            self.transform,
            distribution
        )

    def _create_likelihood(self):
        return DeconvGaussian()

    def _create_input_encoder(self):
        return DeconvInputEncoder(self.dimensions, self.context_size)

    def _create_approximate_posterior(self):

        distribution = StandardNormal((self.dimensions,))

        posterior_transform = self._create_transform(self.context_size, hidden_features=self.hidden_features)

        return HandleInfFlow(
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

    def fit(self, data, val_data=None, show_bar=True, seed=None, raise_bad=False):
        for i in self.iter_fit(data, val_data, show_bar, seed, rewind_on_inf=raise_bad):
            pass

    def checkpoint_fitting(self, optimiser, scheduler):
        previous_state = deepcopy(self.model.state_dict())
        previous_scheduler_state = deepcopy(scheduler.state_dict())
        previous_optimiser_state = deepcopy(optimiser.state_dict())
        return (scheduler, optimiser, previous_state, previous_optimiser_state, previous_scheduler_state)

    def is_bad_step(self, loss):
        bad_model = any(map(lambda x: torch.isnan(x).any().cpu().numpy(), self.model.parameters(recurse=True)))
        return bad_model or not np.isfinite(loss)

    def undo_step(self, epoch, checkpoint):
        print(f'Epoch {epoch}: model parameters got updated to NaN or resulting in invalid loss, rolling back and slowing down')
        scheduler, optimiser, previous_state, previous_optimiser_state, previous_scheduler_state = checkpoint
        self.model.load_state_dict(previous_state)
        optimiser.load_state_dict(previous_optimiser_state)
        scheduler.load_state_dict(previous_scheduler_state)
        scheduler._reduce_lr(epoch)
        scheduler.cooldown_counter = scheduler.cooldown
        scheduler.num_bad_epochs = 0
        scheduler._last_lr = [group['lr'] for group in scheduler.optimizer.param_groups]

    def iter_fit(self, data, val_data=None, show_bar=True, seed=None, use_cuda=False, num_workers=4,
                 rewind_on_inf=False, return_kl_logl=False):
        optimiser = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.lr
        )
        g, worker_init_fn = None, None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            worker_init_fn = seed_worker


        loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            generator=g,
            worker_init_fn=worker_init_fn
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode='max',
            factor=0.8,
            patience=20,
            verbose=True,
            threshold=1e-6
        )
        bar = tqdm(range(self.epochs), disable=not show_bar, unit='epochs', smoothing=1)
        train_loss = 0.0
        val_loss = 0.0
        kl_multiplier = 1.
        checkpoint = self.checkpoint_fitting(optimiser, scheduler)

        for iepoch in bar:
            try:
                self.model.train()
                train_loss = 0.0
                logls = 0.
                kls = 0.
                torch.set_default_tensor_type(torch.FloatTensor)

                for j, d in enumerate(loader):
                    d = [a.to(self.device) for a in d]

                    optimiser.zero_grad()
                    if use_cuda:
                        torch.set_default_tensor_type(torch.cuda.FloatTensor)

                    if self.use_iwae:
                        objective, (logl, kl) = self.model.log_prob_lower_bound(
                            d,
                            num_samples=self.n_samples,
                            kl_multiplier=kl_multiplier,
                            separate=True
                        )
                    else:
                        objective, (logl, kl) = self.model.stochastic_elbo(
                            d,
                            num_samples=self.n_samples,
                            kl_multiplier=kl_multiplier,
                            separate=True
                        )
                    torch.set_default_tensor_type(torch.FloatTensor)

                    minibatch_scaling = data.X.shape[0] / d[0].shape[0]
                    objective = torch.clamp(objective, -1e+30, 1e+30).sum()
                    train_loss += torch.sum(objective * minibatch_scaling).item()
                    logls += torch.sum(logl * minibatch_scaling).item()
                    kls += torch.sum(kl * minibatch_scaling).item()
                    loss = -1 * torch.sum(objective * minibatch_scaling)  # changed to sum with minibatch scaling rather than mean
                    loss.backward()
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip_norm
                        )
                    optimiser.step()
                logls /= len(data)
                kls /= len(data)

                if self.is_bad_step(train_loss):
                    if not rewind_on_inf:
                        raise ValueError(f"Bad loss or network parameters")
                    self.undo_step(iepoch, checkpoint)
                    checkpoint = self.checkpoint_fitting(optimiser, scheduler)
                    continue
                else:
                    checkpoint = self.checkpoint_fitting(optimiser, scheduler)
                val_loss = None
                if val_data:
                    val_loss = self.score_batch(
                        val_data,
                        log_prob=self.use_iwae,
                        num_samples=self.n_samples,
                        use_cuda=use_cuda,
                        num_workers=num_workers
                    ) * minibatch_scaling
                    bar.desc = f'loss[train|val], [logl|kl]: {train_loss:.4f} | {val_loss:.4f}, {logls:.4f} | {kls:.4f}'
                    scheduler.step(val_loss)
                else:
                    bar.desc = f'loss[train], [logl|kl]: {train_loss:.4f}, {logls:.4f} | {kls:.4f}'
                    scheduler.step(train_loss)
                if return_kl_logl:
                    yield self, train_loss, val_loss, logls, kls
                else:
                    yield self, train_loss, val_loss
            except KeyboardInterrupt:
                if return_kl_logl:
                    yield self, train_loss, val_loss, logls, kls
                else:
                    yield self, train_loss, val_loss
                return

    def score(self, data, log_prob=False, num_samples=None, use_cuda=False):
        
        if not num_samples:
            num_samples = self.n_samples

        with torch.no_grad():
            self.model.eval()

            if use_cuda:
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
            if log_prob:
                return self.model.log_prob_lower_bound(data, num_samples=num_samples)
            else:
                return self.model.stochastic_elbo(data, num_samples=num_samples)
            torch.set_default_tensor_type(torch.FloatTensor)

    def score_batch(self, dataset, log_prob=False, num_samples=None, use_cuda=False, num_workers=4):
        loader = data_utils.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )
        score = 0.0

        for j, d in enumerate(loader):
            d = [a.to(self.device) for a in d]
            score += torch.sum(self.score(d, log_prob, num_samples, use_cuda)).item()

        return score

    def sample_prior(self, num_samples, device=torch.device('cpu')):
        with torch.no_grad():
            self.model.eval()
            return minibatch_sample(
                self.model._prior.sample,
                num_samples,
                self.dimensions,
                self.batch_size,
                device
            )
        
    def sample_posterior(self, x, num_samples, device=torch.device('cpu')):
        with torch.no_grad():
            self.model.eval()
            context = self.model._inputs_encoder(x)
            return minibatch_sample(
                self.model._approximate_posterior.sample,
                num_samples,
                self.dimensions,
                self.batch_size,
                device,
                context=context
            )
        
    def _resample_posterior(self, x, num_samples, context):
        
        samples, log_q_z = self.model._approximate_posterior.sample_and_log_prob(
            num_samples,
            context=context
        )
        samples = utils.merge_leading_dims(samples, num_dims=2)
        log_q_z = utils.merge_leading_dims(log_q_z, num_dims=2)

        # Compute log prob of latents under the prior.
        log_p_z = self.model._prior.log_prob(samples)

        # Compute log prob of inputs under the decoder,
        x = [utils.repeat_rows(_x, num_reps=num_samples) for _x in x]
        log_p_x = self.model._likelihood.log_prob(x, context=samples)

        # Compute ELBO.
        log_w = log_p_x + log_p_z - log_q_z
        log_w = utils.split_leading_dim(log_w, [-1, num_samples])
        log_w[torch.isnan(log_w)] = -torch.inf
        log_w -= torch.logsumexp(log_w, dim=-1)[:, None]
        log_w[~torch.isfinite(log_w)] = torch.finfo().min
        
        samples = utils.split_leading_dim(samples, [-1, num_samples])
        idx = torch.distributions.Categorical(logits=log_w).sample([num_samples])
        
        return samples[
            torch.arange(samples.shape[0], device=self.device)[:, None, None],
            idx.T[:, :, None],
            torch.arange(self.dimensions, device=self.device)[None, None, :]
        ]
    
    def resample_posterior(self, x, num_samples, device=torch.device('cpu')):
        with torch.no_grad():
            self.model.eval()
            context = self.model._inputs_encoder(x)
            
            return minibatch_sample(
                self._resample_posterior,
                num_samples,
                self.dimensions,
                self.batch_size,
                device,
                context=context,
                x=x
            )
        
        
        
       
            
    