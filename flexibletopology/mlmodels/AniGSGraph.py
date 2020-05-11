import os
import torch
from torch import nn
import numpy as np

from flexibletopology.mlmodels.GSGraph import GSGraph
from torch.jit import Final
import torchani


class AniGSGraph(nn.Module):

    TORCHANI_PATH: Final[str]
    TORCHANI_PARAMS_FILE: Final[str]
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

        self.TORCHANI_PATH = os.path.dirname(os.path.realpath(torchani.__file__))
        self.TORCHANI_PARAMS_FILE = '../torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params'

    def ani_aev(self, coordinates):

        num_atoms = coordinates.shape[0]
        atom_types = ''
        for i in range(num_atoms):
            atom_types += 'C'

        const_file = os.path.join(self.TORCHANI_PATH,
                                  self.TORCHANI_PARAMS_FILE)

        consts = torchani.neurochem.Constants(const_file)
        aev_computer = torchani.AEVComputer(**consts)
        species = consts.species_to_tensor(atom_types).unsqueeze(0)
        _, aev_signals = aev_computer((species,
                                       coordinates.unsqueeze(0)))

        return aev_signals

    def forward(self, coordinates, signals):

        aev_signals = self.ani_aev(coordinates)

        signals = torch.cat((aev_signals.squeeze(0),
                             signals), 1)
        features = self.gsg_model(coordinates, signals)

        return features
