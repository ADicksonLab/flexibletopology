import os
import os.path as osp
import numpy as np
import pickle as pkl
from itertools import product

import torch
from torch.autograd import Variable

import torchani

from flexibletopology.mlmodels.AniGSGraph import AniGSGraph
import warnings

warnings.filterwarnings("ignore")

MODEL_SAVE_PATH = 'inputs/ani_gsg_model.pt'
DATASET_NAME = 'inputs/openchem_3D_8_110.pkl'



PARAMS = {'wavelet_steps': [1, 2, 3, 4, 5, 6, 7, 8],
          'scf_flags': [(True, True, True), (True, True, False),
                       (True, False, True), (False, True, True)]}



def AniGSG(wavelet_num_steps, scf_flags):
    #read molecule properties from the file
    with open(DATASET_NAME, 'rb') as pklf:
        data = pkl.load(pklf)


    nans_count = 0
    #nm
    radial_cutoff = 0.75
    scf_flags= (True, True, False)

    AniGSG_model = AniGSGraph(wavelet_num_steps=wavelet_num_steps,
                    radial_cutoff=radial_cutoff,
                    scf_flags=scf_flags)

    for mol_idx in range(len(data['molid'])):
        coordinates = np.copy(data['coords'][mol_idx]) / 10
        signals = np.copy(data['gaff_signals_notype'][mol_idx])

        coordinates = torch.from_numpy(coordinates)
        coordinates.requires_grad = True

        signals = torch.from_numpy(signals)
        signals.requires_grad = True

        ani_gsg_features = AniGSG_model(coordinates, signals)

        nans = torch.where(torch.isnan(ani_gsg_features.squeeze(1))==True)[0]
        if nans.shape[0] > 0 :
            nans_count += 1

    return nans_count


if __name__=="__main__":

    print("The total number of molecules is 250")
    gsg_params = product(PARAMS['wavelet_steps'], PARAMS['scf_flags'])
    for wavelet_steps, scf_flags in gsg_params:

        nans_count = AniGSG(wavelet_steps, scf_flags)
        print(f'Model with wavelet_steps {wavelet_steps} and scf flags of {scf_flags} has {nans_count} molecules with nan fetures')
