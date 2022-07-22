from typing import List

import torch
from nflows.transforms import Transform, Sigmoid, InputOutsideDomain, CompositeTransform, InverseTransform
from nflows.utils import torchutils
from torch.nn import functional as F
import numpy as np

class BatchedTransform(Transform):
    def inverse(self, inputs, context=None):
        x, logabsdet = self._inverse(inputs, context)
        return x, torch.sum(logabsdet, dim=-1)

    def forward(self, inputs, context=None):
        x, logabsdet = self._forward(inputs, context)
        return x, torch.sum(logabsdet, dim=-1)


class LowerBound(BatchedTransform):
    def __init__(self, lower, eps=1e-6) -> None:
        super().__init__()
        self.lower = lower
        self.eps = eps

    def _inverse(self, inputs, context=None):
        """
        from [l, +oo] -> [1, 0]
        """
        y = torch.exp(self.lower - inputs)
        return torch.clamp(y, self.eps, 1-self.eps), self.lower - inputs # |dy/dx| = e^(l-x)

    def _forward(self, inputs, context=None):
        """
        from [1, 0] -> [l, +oo]
        """
        x = -torch.log(torch.clamp(inputs, self.eps, 1-self.eps))
        return self.lower + x, x  # |d/dx| is -log(|inputs|) but inputs is always positive, so simplify


class UpperBound(BatchedTransform):
    def __init__(self, upper, eps=1e-6) -> None:
        super().__init__()
        self.upper = upper
        self.eps = eps

    def _inverse(self, inputs, context=None):
        """
        from [-oo, u] -> [0, 1]
        """
        x = torch.exp(inputs - self.upper)
        return torch.clamp(x, self.eps, 1-self.eps), inputs - self.upper

    def _forward(self, inputs, context=None):
        """
        from [0, 1] -> [-oo, u]
        """
        y = torch.log(torch.clamp(inputs, self.eps, 1-self.eps))
        return y + self.upper, -y


class TwoSidedBound(BatchedTransform):
    def __init__(self, lower, upper, eps=1e-6):
        super(TwoSidedBound, self).__init__()
        self.lower = lower
        self.upper = upper
        self.scale = upper - lower
        self.eps = eps
        try:
            self.logscale = torch.log(torch.scalar_tensor(self.scale))
        except TypeError:
            self.logscale = torch.log(self.scale)

    def _inverse(self, inputs, context=None):
        """
        from [l, u] -> [0, 1]
        """
        outputs = torch.clamp((inputs - self.lower) / self.scale, self.eps, 1 - self.eps)
        return outputs, -self.logscale

    def _forward(self, inputs, context=None):
        """
        from [0, 1] -> [l, u]
        """
        outputs = (torch.clamp(inputs, self.eps, 1-self.eps) * self.scale) + self.lower
        return outputs, self.logscale


class CompositeBound(Transform):
    def __init__(self, lower, upper, eps=1e-6) -> None:
        super().__init__()
        self.lower = torch.as_tensor(lower)
        self.upper = torch.as_tensor(upper)
        self.scale = self.upper - self.lower
        self.eps = eps
        self.is_lower_only = ~torch.isfinite(self.upper) & torch.isfinite(self.lower)
        self.is_upper_only = torch.isfinite(self.upper) & ~torch.isfinite(self.lower)
        self.is_both = torch.isfinite(self.upper) & torch.isfinite(self.lower)
        self.lower_only = LowerBound(self.lower[self.is_lower_only])
        self.upper_only = UpperBound(self.upper[self.is_upper_only])
        self.both = TwoSidedBound(self.lower[self.is_both], self.upper[self.is_both], eps)

    def _transform_template(self, which, inputs, context):
        lower = getattr(self.lower_only, which)(inputs[:, self.is_lower_only], context)
        upper = getattr(self.upper_only, which)(inputs[:, self.is_upper_only], context)
        both = getattr(self.both, which)(inputs[:, self.is_both], context)
        outputs, logabsdet = inputs.clone(), torch.zeros(inputs.shape[0], dtype=inputs.dtype)
        outputs[:, self.is_lower_only] = lower[0]
        outputs[:, self.is_upper_only] = upper[0]
        outputs[:, self.is_both] = both[0]
        logabsdet = lower[1] + upper[1] + both[1]  # we dont need to sum here since torch.sum used above expands into a n-sized tensor
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        return self._transform_template('inverse', inputs, context)

    def forward(self, inputs, context=None):
        return self._transform_template('forward', inputs, context)

#
# class SkipSigmoid(Sigmoid):
#     def __init__(self, skip_dims, temperature=1, eps=1e-6, learn_temperature=False):
#         """
#         A sigmoid, but the skip_dims boolean mask tells it not to transform the given dimension at all
#         """
#         super().__init__(temperature, eps, learn_temperature)
#         self.skip_dims = skip_dims
#
#     def _forward(self, inputs, context=None):
#         inputs = self.temperature * inputs
#         outputs = torch.sigmoid(inputs)
#         logabsdet = torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs)
#         return outputs, logabsdet
#
#     def _inverse(self, inputs, context=None):
#         if torch.min(inputs[:, ~self.skip_dims]) < 0 or torch.max(inputs[:, ~self.skip_dims]) > 1:
#             raise InputOutsideDomain()
#
#         inputs = torch.clamp(inputs, self.eps, 1 - self.eps)
#
#         outputs = (1 / self.temperature) * (torch.log(inputs) - torch.log1p(-inputs))
#         logabsdet = torch.log(self.temperature) - F.softplus(-self.temperature * outputs) - F.softplus(self.temperature * outputs)
#         return outputs, -logabsdet
#
#     def forward(self, inputs, context=None):
#         x, logabsdet = self._forward(inputs[:, ~self.skip_dims], context)
#         result = inputs.clone()
#         result[:, ~self.skip_dims] = x  # no need to do this for logabsdet since we sum and it would be 0
#         return result, torchutils.sum_except_batch(logabsdet)
#
#     def inverse(self, inputs, context=None):
#         x, logabsdet = self._inverse(inputs[:, ~self.skip_dims], context)
#         result = inputs.clone()
#         result[:, ~self.skip_dims] = x  # no need to do this for logabsdet since we sum and it would be 0
#         return result, torchutils.sum_except_batch(logabsdet)


class BoundSpace(CompositeTransform):
    def __init__(self, lower_bounds, upper_bounds, eps):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.skip_dims = ~(torch.isfinite(self.lower_bounds) | torch.isfinite(self.upper_bounds))
        self.skip_all = bool(torch.all(self.skip_dims).numpy())
        super().__init__([Sigmoid(eps=eps), CompositeBound(self.lower_bounds[~self.skip_dims], self.upper_bounds[~self.skip_dims], eps)])

    def forward(self, inputs, context=None):
        if self.skip_all:
            return inputs, torch.zeros(inputs.shape[0], dtype=inputs.dtype)
        x, logabsdet = super().forward(inputs[:, ~self.skip_dims], context)
        result = inputs.clone()
        result[:, ~self.skip_dims] = x  # no need to do this for logabsdet since we sum and it would be 0
        return result, logabsdet

    def inverse(self, inputs, context=None):
        if self.skip_all:
            return inputs, torch.zeros(inputs.shape[0], dtype=inputs.dtype)
        x, logabsdet = super().inverse(inputs[:, ~self.skip_dims], context)
        result = inputs.clone()
        result[:, ~self.skip_dims] = x  # no need to do this for logabsdet since we sum and it would be 0
        return result, logabsdet


# class SigmoidBound(Sigmoid):
#     def __init__(self, lower, upper, eps=1e-6):
#         super().__init__(1., eps, False)
#         self.lower = lower
#         self.upper = upper
#         self.scale = upper - lower
#         if torch.any(torch.log10(self.scale) >= -np.log10(self.eps) / 2).numpy():
#             raise RuntimeError(f"Cannot use bounds which have a larger width than the eps precision")
#         try:
#             self.logscale = torch.log(torch.scalar_tensor(self.scale))
#         except TypeError:
#             self.logscale = torch.log(self.scale)
#
#     def forward(self, inputs, context=None):
#         """
#         from [-oo, +oo] -> [l, u]
#         """
#         inputs = self.temperature * inputs
#         outputs = torch.sigmoid(inputs) * self.scale + self.lower
#         logabsdets = (torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs)) + self.logscale
#         return outputs, torchutils.sum_except_batch(logabsdets)
#
#     def inverse(self, inputs, context=None):
#         """
#         from [l, u] -> [-oo, +oo]
#         """
#         if torch.any(torch.min(inputs) < self.lower) or torch.any(torch.max(inputs) > self.upper):
#             raise InputOutsideDomain()
#
#         inputs = torch.clamp((inputs - self.lower) / self.scale, self.eps, 1 - self.eps)
#
#         outputs = (1 / self.temperature) * (torch.log(inputs) - torch.log1p(-inputs))
#         logabsdets = torch.log(self.temperature) - F.softplus(-self.temperature * outputs) - F.softplus(self.temperature * outputs)
#         logabsdets += self.logscale
if __name__ == '__main__':
    # transforms for
    # [-oo, +oo] -> [-oo, u]
    # [-oo, +oo] -> [l, +oo]
    # [-oo, +oo] -> [l, u]
    import numpy as np
    import matplotlib.pyplot as plt
    eps = 1e-6
    lower_bounds = np.array([-np.inf, 0])
    upper_bounds = np.array([10., 10000.])

    bound = BoundSpace(lower_bounds, upper_bounds, eps)
    # bound = SkipSigmoid(torch.tensor([False, True]))
    # bound = CompositeTransform([Sigmoid(eps=eps), CompositeBound(lower_bounds, upper_bounds, eps)])

    fig, axs = plt.subplots(2)
    Xdata = np.random.normal(0, 2, (1000, 2))
    Ydata = bound.forward(torch.as_tensor(Xdata))[0]
    Xdata_ = bound.inverse(Ydata)[0]
    axs[0].scatter(*Xdata.T, s=1)
    axs[0].scatter(*Xdata_.T.numpy(), s=1)
    axs[1].scatter(*Ydata.T.numpy(), s=1)

    plt.figure()
    delta = (Xdata - Xdata_.numpy()).ravel()
    plt.hist(delta, bins=100)
    plt.title(f"{np.sum(~np.isfinite(delta)):.0f} bad")

    axs[0].set_title('Unbounded data')
    axs[1].set_title('Bounded data')
    plt.show()

