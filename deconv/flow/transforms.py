from typing import List

import torch
from nflows.transforms import Transform, Sigmoid, InputOutsideDomain
from nflows.utils import torchutils
from torch.nn import functional as F
import numpy as np


class SigmoidBound(Sigmoid):
    def __init__(self, lower, upper, lowerpoint99, upperpoint99, eps=1e-6):
        super().__init__(1., eps, False)
        self.lower = lower
        self.upper = upper
        self.lowerpoint99 = lowerpoint99
        self.upperpoint99 = upperpoint99
        self.A = lower
        self.B = self.upper - self.lower
        width = (upper - lower)
        lower99 = lower + (0.99 * width)
        upper99 = upper - (0.99 * width)
        k0 = np.log((width / (lower99 - self.lower) )- 1)
        k = k0 / np.log((width / (upper99 - lower)) - 1)
        self.D = (self.upperpoint99 - (self.lowerpoint99 * k)) / (1 - k)
        self.C = -k0 / (self.upperpoint99 - self.D)
        self.A = torch.as_tensor(self.A)
        self.B = torch.as_tensor(self.B)
        self.C = torch.as_tensor(self.C)
        self.D = torch.as_tensor(self.D)

    def inverse(self, inputs, context=None):
        dtype = inputs.dtype
        inputs = self.C * (inputs - self.D)
        outputs = torch.sigmoid(inputs)
        logabsdet = torchutils.sum_except_batch((
            torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs)
        )  + torch.log(torch.abs(outputs * self.B * self.C)))
        outputs = (self.B * outputs + self.A)
        outputs = torch.as_tensor(outputs, dtype=dtype)
        logabsdet = torch.as_tensor(logabsdet, dtype=dtype)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        dtype = inputs.dtype
        if torch.any(torch.min(inputs, 0)[0] < torch.as_tensor(self.lower)) or torch.any(torch.max(inputs, 0)[0] > torch.as_tensor(self.upper)):
            raise InputOutsideDomain()
        inputs = torch.clamp(inputs, torch.as_tensor(self.lower) + self.eps, torch.as_tensor(self.upper) - self.eps)
        inputs = (inputs - self.A) / self.B
        outputs = (1 / self.temperature) * (torch.log(inputs) - torch.log1p(-inputs))
        outputs = (outputs / self.C) + self.D
        logabsdet = -torchutils.sum_except_batch((
            torch.log(self.temperature)
            - F.softplus(-self.temperature * outputs)
            - F.softplus(self.temperature * outputs)
        )  + torch.log(torch.abs(self.C / self.B)))
        outputs = torch.as_tensor(outputs, dtype=dtype)
        logabsdet = torch.as_tensor(logabsdet, dtype=dtype)
        return outputs, logabsdet

#
#
# class PositiveOnly(Transform):
#     def __init__(self, dimensions: List[int], scales: List[float] = None) -> None:
#         """
#         Apply exponential to `dimensions` of the inputs leaving the other dimensions untouched
#         This transforms from the [-oo,+oo] -> [0, +oo] / exp(scale)
#         """
#         super().__init__()
#         if scales is None:
#             self.scales = torch.ones(len(dimensions))
#         else:
#             if len(scales) != len(dimensions):
#                 raise ValueError(f"dimensions must be the same length as scales")
#             self.scales = torch.as_tensor(scales)
#         self.dimensions = dimensions
#
#     def inverse(self, inputs, context=None):
#         outputs = torch.clone(inputs)
#         # outputs[:, self.dimensions] = torch.exp(torch.clamp(inputs[:, self.dimensions] / self.scales, max=50))
#         outputs[:, self.dimensions] = torch.clamp(outputs[:, self.dimensions], max=1e+9) ** 2.
#         logabsdet = torch.sum(torch.log(torch.abs(2 * outputs[:, self.dimensions])))
#         # logabsdet = torch.sum(torch.abs(inputs[:, self.dimensions]) / self.scales, dim=1)
#         return outputs, logabsdet
#
#     def forward(self, inputs, context=None):
#         outputs = torch.clone(inputs)
#         outputs[:, self.dimensions] = torch.sqrt(inputs[:, self.dimensions]) #torch.log(inputs[:, self.dimensions]) * self.scales
#         # logabsdet = torch.sum(-torch.log(torch.abs(inputs[:, self.dimensions] / self.scales)), dim=1)
#         logabsdet = torch.sum(torch.log(torch.abs(0.5 * inputs[:, self.dimensions])), dim=1)
#         return outputs, logabsdet
