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

DATASET_NAME = 'inputs/openchem_3D_8_110.pkl'

PARAMS = {'wavelet_steps': [4, 5, 6, 7, 8],
          'scf_flags': [(True, True, True), (True, True, False),
                       (True, False, True), (False, True, True)]}

def test_features(wavelet_num_steps, scf_flags):
    #read molecule properties from the file
    with open(DATASET_NAME, 'rb') as pklf:
        data = pkl.load(pklf)


    nans_count = 0
    #nm
    radial_cutoff = 0.52

    AniGSG_model = AniGSGraph(wavelet_num_steps=wavelet_num_steps,
                    radial_cutoff=radial_cutoff,
                    scf_flags=scf_flags)

    for mol_idx in range(len(data['molid'])):
        coordinates = np.copy(data['coords'][mol_idx])
        signals = np.copy(data['gaff_signals_notype'][mol_idx])

        #coordinates are in nm
        coordinates = torch.from_numpy(coordinates) / 10
        coordinates.requires_grad = True

        signals = torch.from_numpy(signals)
        signals.requires_grad = True

        ani_gsg_features = AniGSG_model(coordinates, signals)

        nans = torch.where(torch.isnan(ani_gsg_features.squeeze(1))==True)[0]

        if nans.shape[0] > 0 :
            nans_count += 1

    return nans_count


def test_gradients(wavelet_num_steps, scf_flags):
    #read molecule properties from the file
    with open(DATASET_NAME, 'rb') as pklf:
        data = pkl.load(pklf)


    nans_count = 0
    #nm
    radial_cutoff = 0.52
    AniGSG_model = AniGSGraph(wavelet_num_steps=wavelet_num_steps,
                    radial_cutoff=radial_cutoff,
                    scf_flags=scf_flags)

    for mol_idx in range(len(data['molid'])):
        coordinates = np.copy(data['coords'][mol_idx])
        signals = np.copy(data['gaff_signals_notype'][mol_idx])

        coordinates = torch.from_numpy(coordinates) / 10
        coordinates.requires_grad = True

        signals = torch.from_numpy(signals)
        signals.requires_grad = True

        ani_gsg_features = AniGSG_model(coordinates, signals)

        loss_fun = torch.nn.MSELoss()
        loss = loss_fun(ani_gsg_features, torch.rand_like(ani_gsg_features))
        loss.backward()
        coord_grad = coordinates.grad
        nans = torch.where(torch.isnan(coord_grad)==True)[0]
        if nans.shape[0] > 0 :
            nans_count += 1
    return nans_count



if __name__=="__main__":

    wavelet_steps = 8
    scf_flags = (True, True, True)
    print("The total number of molecules is 250")
    nans_count = test_features(wavelet_steps, scf_flags)
    print(f'Model with wavelet_steps {wavelet_steps} and scf flags of {scf_flags} has {nans_count} molecules with nan features')

    nans_count = test_gradients(wavelet_steps, scf_flags)
    print(f'Model with wavelet_steps {wavelet_steps} and scf flags of {scf_flags} has {nans_count} molecules with nan coord_grad')
