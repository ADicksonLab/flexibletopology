import torch
from torch import nn
import numpy as np


class GravPotential(nn.Module):

    """
    Implements a flat-bottomed gravitational potential that is equal to
    1/2 * k * (r-r0)**2 when r > r0 and 0 otherwise.

    k is specified by the forceConstant argument in kcal/mol
    r0 is specified by the radius argument in nm
    """

    def __init__(self, forceConstant=20, radius=1.0):
        super(GravPotential, self).__init__()
        self.is_trainable = False
        self.k = forceConstant
        self.r0 = radius
        self.r0sq = radius**2

    def forward(self, positions, signals):

        # center of geometry
        cog = torch.mean(positions,axis=0)

        cog_dists_sq = torch.sum(torch.square(positions - cog),axis=1)
        cog_dists = torch.sqrt(cog_dists_sq)

        en = torch.where(cog_dists > self.r0,
                         0.5 * self.k * torch.square(cog_dists - self.r0),
                         torch.zeros_like(cog_dists))

        return en
