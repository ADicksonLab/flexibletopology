import pytest

import numpy as np
from numpy import array_equal as eq
import scipy.stats as sci_stats

import flexibletopology.utils.stats as fltop_stats

import torch

torch.manual_seed(11)

AXIS = 1
BIAS = False
MOMENT = 4
PRECISION = 3
DATA = torch.rand(6, 4, 3)


def test_skew():

    fltop_skew = fltop_stats.skew(DATA, axis=AXIS, bias=BIAS)
    sci_skew = sci_stats.skew(DATA.numpy(), axis=AXIS, bias=BIAS)
    assert eq(np.round(fltop_skew.numpy(), PRECISION),
              np.round(sci_skew, PRECISION))

def test_kurtosis():
    fltop_kurtosis = fltop_stats.kurtosis(DATA, axis=AXIS, bias=BIAS)
    sci_kurtosis = sci_stats.kurtosis(DATA.numpy(), axis=AXIS, bias=BIAS)
    assert eq(np.round(fltop_kurtosis.numpy(), PRECISION),
              np.round(sci_kurtosis, PRECISION))


def test_moment():
    fltop_moment = fltop_stats.moment(DATA, moment=MOMENT, axis=AXIS)
    sci_moment = sci_stats.moment(DATA.numpy(), moment=MOMENT, axis=AXIS)
    assert eq(np.round(fltop_moment.numpy(), PRECISION),
              np.round(sci_moment, PRECISION))
