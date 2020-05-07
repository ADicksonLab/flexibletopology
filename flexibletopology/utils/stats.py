import torch
import numpy as np

@torch.jit.script
def size_helper(a, axis):
    return torch.tensor(a.shape[axis])

def distance_matrix(x):
    return torch.norm(x[:, None] - x, dim=2, p=2)


def adjacency_matrix(positions, radial_cutoff):
    dist = distance_matrix(positions)
    dist = torch.where(dist>radial_cutoff, torch.tensor(0.0, dtype=dist.dtype),
                        0.5 * torch.cos(np.pi * dist/radial_cutoff) + 0.5)
    dist.fill_diagonal_(0.0)
    return dist



def moment(a, moment=1, axis=0):
    if moment == 1:
        # By definition the first moment about the mean is 0.
        shape = list(a.shape)
        del shape[axis]
        if shape:
            # return an actual array of the appropriate shape
            return torch.zeros(shape)
        else:
            # the input was 1D, so return a scalar instead of a rank-0 array
            return torch.tensor(0.0)
    else:
        # Exponentiation by squares: form exponent sequence
        n_list = [moment]
        current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n-1)/2
            else:
                current_n /= 2
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        a_zero_mean = a - a.mean(axis).unsqueeze(axis)
        if n_list[-1] == 1:
            s = a_zero_mean
        else:
            s = a_zero_mean**2

        # Perform multiplications
        for n in n_list[-2::-1]:
            s = s**2
            if n % 2:
                s *= a_zero_mean
        return s.mean(axis)

def skew(a, axis=0, bias=True):
    n = a.shape[axis]
    m2 = moment(a, 2, axis)
    m3 = moment(a, 3, axis)
    vals = torch.where(m2 == 0.0, torch.tensor(0.0, dtype=m2.dtype), m3 / m2**1.5)

    if not bias and size_helper(a, torch.tensor(axis)) > 2:
        n =  size_helper(a, torch.tensor(axis))
        vals = torch.where(m2 > 0,
                           torch.sqrt(torch.as_tensor((n-torch.tensor(1))*n, dtype=m2.dtype))/(n-torch.tensor(2))*m3/m2**1.5, vals)

    if vals.ndim == 0:
        return vals.item()

    return vals

#impliment kurtosis using the kurtosis code from scipy for two dimensional
#array

def kurtosis(a, axis=0, fisher=True, bias=True):
    m2 = moment(a, 2, axis)
    m4 = moment(a, 4, axis)

    vals = torch.where(m2 == 0.0, torch.tensor(0.0, dtype=m2.dtype), m4 / m2**2.0)

    if not bias and size_helper(a, torch.tensor(axis))> 3:
        n =  size_helper(a, torch.tensor(axis))
        vals = torch.where(m2 > 0,
                           torch.tensor(1.0)/(n-2)/(n-3)*((n*n-1.0)*m4/m2**2.0-3*(n-1)**2.0)+torch.tensor(3.0),
                        vals)
    if vals.ndim == 0:
        return vals.item()

    return vals - 3 if fisher else vals
