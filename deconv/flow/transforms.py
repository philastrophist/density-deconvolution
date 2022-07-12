from typing import List

import torch
from nflows.transforms import Transform, Sigmoid, InputOutsideDomain, CompositeTransform
from nflows.utils import torchutils
from torch.nn import functional as F
import numpy as np

class LowerBound(Transform):
    def __init__(self, lower, eps=1e-6) -> None:
        super().__init__()
        self.lower = lower
        self.eps = eps

    def inverse(self, inputs, context=None):
        """
        from [l, +oo] -> [1, 0]
        """
        y = inputs - self.lower
        outputs = y / (1 + y)
        return torch.clamp(outputs, self.eps, 1-self.eps), -2 * torch.log(1 + y)

    def forward(self, inputs, context=None):
        """
        from [1, 0] -> [l, +oo]
        """
        x = inputs / (1 - inputs)
        return x + self.lower, -2 * torch.log(1 - inputs)


class UpperBound(Transform):
    def __init__(self, upper, eps=1e-6) -> None:
        super().__init__()
        self.upper = upper
        self.eps = eps

    def inverse(self, inputs, context=None):
        """
        from [-oo, u] -> [0, 1]
        """
        y = self.upper - inputs
        outputs = y / (1 + y)
        return torch.clamp(outputs, self.eps, 1-self.eps), -2 * torch.log(1 + y)

    def forward(self, inputs, context=None):
        """
        from [0, 1] -> [-oo, u]
        """
        x = inputs / (1 - inputs)
        return self.upper - x, -2 * torch.log(1 - inputs)


class TwoSidedBound(Transform):
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

    def inverse(self, inputs, context=None):
        """
        from [-oo, +oo] -> [0, 1]
        """
        outputs = torch.clamp((inputs - self.lower) / self.scale, self.eps, 1 - self.eps)
        return outputs, -self.logscale

    def forward(self, inputs, context=None):
        """
        from [0, 1] -> [-oo, +oo]
        """
        outputs = (inputs * self.scale) + self.lower
        return outputs, self.logscale


class CompositeBound(Transform):
    def __init__(self, lower, upper, eps=1e-6) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.scale = upper - lower
        self.eps = eps
        self.is_lower_only = torch.isfinite(self.upper) & ~torch.isfinite(self.lower)
        self.is_upper_only = ~torch.isfinite(self.upper) & torch.isfinite(self.lower)
        self.is_both = torch.isfinite(self.upper) & torch.isfinite(self.lower)
        if torch.any((torch.log10(self.scale) >= -np.log10(self.eps) / 2) & ~self.is_both).numpy():
            raise RuntimeError(f"Cannot use two-sided bounds which have a larger width than the eps precision")
        self.lower_only = LowerBound(self.lower)
        self.upper_only = UpperBound(self.upper)
        self.both = TwoSidedBound(self.lower, self.upper, eps)

    def inverse(self, inputs, context=None):
        lower = self.lower_only.inverse(inputs, context)
        upper = self.upper_only.inverse(inputs, context)
        both = self.both.inverse(inputs, context)
        outputs = torch.where(self.is_both, both[0], torch.where(self.is_lower_only, lower[0], upper[0]))
        logabsdet = torch.where(self.is_both, both[1], torch.where(self.is_lower_only, lower[1], upper[1]))
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        lower = self.lower_only.forward(inputs, context)
        upper = self.upper_only.forward(inputs, context)
        both = self.both.forward(inputs, context)
        outputs = torch.where(self.is_both, both[0], torch.where(self.is_lower_only, lower[0], upper[0]))
        logabsdet = torch.where(self.is_both, both[1], torch.where(self.is_lower_only, lower[1], upper[1]))
        return outputs, logabsdet

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
#         return outputs, -torchutils.sum_except_batch(logabsdets)

if __name__ == '__main__':
    # transforms for
    # [-oo, +oo] -> [-oo, u]
    # [-oo, +oo] -> [l, +oo]
    # [-oo, +oo] -> [l, u]

    eps = 1e-6
    bounds = torch.tensor([
        [0, 1e+1],
        [-0.1, 1e+6]
    ])
    bound = CompositeTransform([Sigmoid(eps=eps), CompositeBound(bounds[:, 0], bounds[:, 1], eps)])


    Xdata = torch.tensor([[0.001, 0.], [0, 1], [1., 2.]])
    print(bound.forward(bound.inverse(Xdata)[0])[0].numpy())
    # print(bound.inverse(Xdata)[0].numpy())