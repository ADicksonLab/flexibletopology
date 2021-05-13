import sys
import torch
import numpy as np
from torch import Tensor
from typing import List


def distance_matrix(x: Tensor) -> Tensor:
    return torch.norm(x[:, None] - x, dim=2, p=2)


def adjacency_matrix(positions: Tensor, radial_cutoff: float) -> Tensor:
    dist = distance_matrix(positions)
    dist = torch.where(dist > radial_cutoff,
                       torch.tensor(0.0, dtype=dist.dtype,
                                    device=positions.device),
                       0.5 * torch.cos(np.pi * dist/radial_cutoff) + 0.5)
    dist.fill_diagonal_(0.0)
    return dist


def moment(a: Tensor, moment: int = 1, dim: int = 0) -> Tensor:
    if moment == 0:
        # When moment equals 0, the result is 1, by definition.
        shape = list(a.shape)
        del shape[dim]
        if len(shape) > 0:
            # return an actual array of the appropriate shape
            return torch.ones(shape, device=a.device)
        else:
            # the input was 1D, so return a scalar instead of a rank-0 array
            return torch.tensor(1.0, device=a.device)

    elif moment == 1:
        # By definition the first moment about the mean is 0.
        shape = list(a.shape)
        del shape[dim]
        if len(shape) > 0:
            # return an actual array of the appropriate shape
            return torch.zeros(shape)
        else:
            # the input was 1D, so return a scalar instead of a rank-0 array
            return torch.tensor(0.0, device=a.device)
    else:
        # Exponentiation by squares: form exponent sequence

        n_list: List[int] = [moment]

        current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = int((current_n - 1)/2)
            else:
                current_n = int(current_n / 2)
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        a_zero_mean = a - a.mean(dim).unsqueeze(dim)
        if n_list[-1] == 1:
            s = a_zero_mean.clone()
        else:
            s = a_zero_mean**2

        # Perform multiplications
        n_list.reverse()
        del n_list[0]
        for n in n_list:
            s = s**2
            if n % 2:
                s *= a_zero_mean

        return s.mean(dim)

# unnormalized moments


def skew(a: Tensor, dim: int = 0, bias: bool = True) -> Tensor:
    return moment(a, 3, dim)


# unnormalized moments
def kurtosis(a: Tensor, dim: int = 0, fisher: bool = True,
             bias: bool = True) -> Tensor:

    return moment(a, 4, dim)
