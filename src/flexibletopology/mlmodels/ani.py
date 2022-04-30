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
from .aev import AEVComputer


class Ani(nn.Module):
    """Create a TorchANI model used in Flexible Topology simulations.

    You can find more detailed information in the paper "'ANI-1: an
    extensible neural network potential with DFT accuracy at force
    field computational cost".  It takes the path to the configure
    file, including the TorchANI model parameters and platform type
    as inputs and then generates atomic species of the same type.
    These inputs create a TorchANI model that calculates the radial
    and angular AEVs.

    """

    def __init__(self, platform: str = '', consts_file: str = ''):
        """Constructor for the ANI model.  Read the parameters onfiguration
        file and create and instance of "AEVComputer".

        Args:

        platform (str, optional): The platform is used to create
        the TorchANI model. The accepted values are cpu, cuda or
        opencl. Defaults to ''.

        consts_file (str, optional): The path to the
        file including ANI parameters. You can find example files in
        `resources/ani_params`. Defaults to ''.

        """
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
            self.aev_computer = AEVComputer(**cuda_consts)
        else:
            self.aev_computer = AEVComputer(**consts)

    def forward(self, coordinates: Tensor, charges: Tensor) -> Tensor:
        """Calls the ANI model to calculate AEVs.

        The coordinates must be in ``(N, 3)`` shape and charges are in
        the shape of ``(N)`` where "N" is the number of inputs.

        Args:
            coordinates (Tensor): The coordinates of the molecule in 3D
            charges (Tensor): The partial charge on atoms

        Returns:
            Tensor: The radial and angular AEVs with the shape
            ``(N, M)`` where ``(N)`` is the number of atoms and ``M``
            depends on the TorcANI model parameters.

        """
        assert len(coordinates.shape) == 2, "coordinates should be rank 2 array"
        assert coordinates.shape[1] == 3, "coordinates are not of 3 dimensions"
        assert len(charges) == 1, "charges are not of 1 dimensions"
        assert coordinates.shape[0] == charges.shape[0], "coordinates and charges must have the same number of atoms"

        species = torch.zeros((1, coordinates.shape[0]),
                              dtype=torch.int64,
                              device=coordinates.device)
        _, aev_signals = self.aev_computer((species,
                                            coordinates.unsqueeze(0),
                                            charges))

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
