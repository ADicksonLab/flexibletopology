import sys
import torch
import numpy as np
from torch import Tensor
from typing import List


def distance_matrix(x:Tensor):
    return torch.norm(x[:, None] - x, dim=2, p=2)


<<<<<<< HEAD
def adjacency_matrix(positions, radial_cutoff:float):
=======
def adjacency_matrix(positions:Tensor, radial_cutoff:float):
>>>>>>> water_test
    dist = distance_matrix(positions)
    dist = torch.where(dist>radial_cutoff, torch.tensor(0.0, dtype=dist.dtype),
                        0.5 * torch.cos(np.pi * dist/radial_cutoff) + 0.5)
    dist.fill_diagonal_(0.0)
    return dist



def moment(a: Tensor, moment: int=1, dim: int=0):


    if moment == 0:
        # When moment equals 0, the result is 1, by definition.
        shape = list(a.shape)
        del shape[dim]
        if len(shape) > 0:
            # return an actual array of the appropriate shape
            return torch.ones(shape)
        else:
            # the input was 1D, so return a scalar instead of a rank-0 array
            return torch.tensor(1.0)

    elif moment == 1:
        # By definition the first moment about the mean is 0.
        shape = list(a.shape)
        del shape[dim]
        if len(shape) > 0:
            # return an actual array of the appropriate shape
            return torch.zeros(shape)
        else:
            # the input was 1D, so return a scalar instead of a rank-0 array
            return torch.tensor(0.0)
    else:
        # Exponentiation by squares: form exponent sequence

        n_list: List[int] = [moment]

        current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = int((current_n - 1)/2)
            else:
                current_n = int(current_n/ 2)
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

def skew(a: Tensor, dim: int=0, bias: bool=True):
    n = a.shape[dim]
    m2 = moment(a, 2, dim)
    m3 = moment(a, 3, dim)

    vals = torch.where(m2 == 0.0,
                       torch.tensor(0.0, dtype=m2.dtype),
                       m3/m2**1.5)
    if not bias and n > 2:
        vals = torch.where(m2 > 0,
                           torch.sqrt(torch.tensor((n-1.0)*n))/(n-2)*m3/m2**1.5,
                           vals)

    return vals


def kurtosis(a: Tensor, dim: int=0, fisher: bool=True, bias:bool=True):
    n = a.shape[dim]
    m2 = moment(a, 2, dim)
    m4 = moment(a, 4, dim)

    vals = torch.where(m2 == 0.0,
                       torch.tensor(0.0, dtype=m2.dtype),
                       m4/m2**2.0)

    if not bias and n>3:
        vals = torch.where(m2 > 0,
                            1.0/(n-2)/(n-3)* \
                           ((n**2-1.0)*m4/m2**2.0-3.0*(n-1.0)**2.0)+3.0,
                            vals)
    if vals.ndim == 0:
        return vals

    return vals - 3 if fisher else vals
