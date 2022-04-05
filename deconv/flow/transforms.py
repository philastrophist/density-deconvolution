from typing import List

import torch
from nflows.transforms import Transform, Sigmoid


class PositiveOnly(Transform):
    def __init__(self, dimensions: List[int], scales: List[float] = None) -> None:
        """
        Apply exponential to `dimensions` of the inputs leaving the other dimensions untouched
        This transforms from the [-oo,+oo] -> [0, +oo] / exp(scale)
        """
        super().__init__()
        if scales is None:
            self.scales = torch.ones(len(dimensions))
        else:
            if len(scales) != len(dimensions):
                raise ValueError(f"dimensions must be the same length as scales")
            self.scales = torch.as_tensor(scales)
        self.dimensions = dimensions

    def inverse(self, inputs, context=None):
        outputs = torch.clone(inputs)
        # outputs[:, self.dimensions] = torch.exp(torch.clamp(inputs[:, self.dimensions] / self.scales, max=50))
        outputs[:, self.dimensions] = torch.clamp(outputs[:, self.dimensions], max=1e+9) ** 2.
        logabsdet = torch.sum(torch.log(torch.abs(2 * outputs[:, self.dimensions])))
        # logabsdet = torch.sum(torch.abs(inputs[:, self.dimensions]) / self.scales, dim=1)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        outputs = torch.clone(inputs)
        outputs[:, self.dimensions] = torch.sqrt(inputs[:, self.dimensions]) #torch.log(inputs[:, self.dimensions]) * self.scales
        # logabsdet = torch.sum(-torch.log(torch.abs(inputs[:, self.dimensions] / self.scales)), dim=1)
        logabsdet = torch.sum(torch.log(torch.abs(0.5 * inputs[:, self.dimensions])), dim=1)
        return outputs, logabsdet
