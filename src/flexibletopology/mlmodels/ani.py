import os
import numpy as np

import torch
from torch import Tensor
from typing import Tuple, Optional, NamedTuple

from torch import nn
from torch.jit import Final
import torchani
from typing import List
from .gsg import GSG


class Ani(nn.Module):

    def __init__(self, platform: str = '', consts_file: str = ''):

        super().__init__()
        self.is_trainable = False
        self.consts_file = consts_file
        self.device = torch.device(platform)

        consts = torchani.neurochem.Constants(self.consts_file)
        cuda_consts = {}
        if platform == 'cuda':
            for key, value in consts.items():
                if torch.is_tensor(value):
                    cuda_consts.update({key: value.to(self.device)})
                else:
                    cuda_consts.update({key: value})
            self.aev_computer = torchani.AEVComputer(**cuda_consts)
        else:
            self.aev_computer = torchani.AEVComputer(**consts)

    def forward(self, coordinates: Tensor) -> Tensor:

        species = torch.zeros((1, coordinates.shape[0]),
                              dtype=torch.int64,
                              device=coordinates.device)
        _, aev_signals = self.aev_computer((species,
                                            coordinates.unsqueeze(0)))

        return aev_signals.squeeze()


class AniGSG(nn.Module):

    def __init__(self, max_wavelet_scale: int = 4, radial_cutoff: float = 0.52,
                 sm_operators: Tuple[bool, bool, bool] = (True, True, True),
                 consts_file: str = '',
                 sd_params: Optional[List[List[float]]] = None):

        super().__init__()
        self.is_trainable = False
        self.consts_file = consts_file
        self.max_wavelet_scale = max_wavelet_scale
        self.radial_cutoff = radial_cutoff
        self.sm_operators = sm_operators
        self.sd_params = sd_params

        self.gsg_model = GSG(max_wavelet_scale=self.max_wavelet_scale,
                             radial_cutoff=self.radial_cutoff,
                             sm_operators=self.sm_operators,
                             sd_params=self.sd_params)

        consts = torchani.neurochem.Constants(self.consts_file)
        self.aev_computer = torchani.AEVComputer(**consts)

    def forward(self, coordinates: Tensor, signals: Tensor) -> Tensor:

        species = torch.ones((1, coordinates.shape[0]), dtype=torch.int64)
        _, aev_signals = self.aev_computer((species,
                                            coordinates.unsqueeze(0)))

        signals = torch.cat((aev_signals.squeeze(0),
                             signals), 1)

        features = self.gsg_model(coordinates, signals)
        return features
