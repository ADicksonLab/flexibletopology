import os
import torch
from torch import nn
import numpy as np

from flexibletopology.mlmodels.GSGraph import GSGraph
from torch.jit import Final
import torchani

from .aev import AEVComputer

class AniGSGraph(nn.Module):

    BASE_DIR: Final[str]
    ANI_PARAMS_FILE: Final[str]
    CONST_FILE: Final[str]

    def __init__(self, wavelet_num_steps=4, radial_cutoff=7.5,
                 scf_flags=(True, True, True)):
        super(AniGSGraph, self).__init__()

        self.is_trainable = False
        self.wavelet_num_steps = wavelet_num_steps
        self.radial_cutoff = radial_cutoff
        self.scf_flags = scf_flags

        self.gsg_model = GSGraph(wavelet_num_steps=self.wavelet_num_steps,
                            radial_cutoff=self.radial_cutoff,
                            scf_flags=self.scf_flags)

        self.BASE_DIR = os.path.dirname(os.path.realpath(__file__))
        self.ANI_PARAMS_FILE = '../resources/ani_params/ani-1ccx_8x_nm.params'

    def ani_aev(self, coordinates):

        num_atoms = coordinates.shape[0]
        atom_types = ''
        for i in range(num_atoms):
            atom_types += 'C'

        const_file = os.path.join(self.BASE_DIR,
                                  self.ANI_PARAMS_FILE)

        consts = torchani.neurochem.Constants(const_file)
        aev_computer = AEVComputer(**consts)
        species = consts.species_to_tensor(atom_types).unsqueeze(0)
        _, angular_aev_signals = aev_computer((species,
                                       coordinates.unsqueeze(0)))

        return angular_aev_signals

    def forward(self, coordinates, signals):

        angular_aev_signals = self.ani_aev(coordinates)


        signals = torch.cat((angular_aev_signals.squeeze(0),
                              signals), 1)

        features = self.gsg_model(coordinates, signals)
        return features
