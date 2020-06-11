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

def test_saved_model(mol_idx):
    #read molecule properties from the file
    with open(DATASET_NAME, 'rb') as pklf:
        data = pkl.load(pklf)

    AniGSG_model =  torch.jit.load(MODEL_SAVE_PATH)

    coordinates = np.copy(data['coords'][mol_idx])
    signals = np.copy(data['gaff_signals_notype'][mol_idx])

    coordinates = torch.from_numpy(coordinates) / 10
    print(coordinates)
    coordinates.requires_grad = True

    signals = torch.from_numpy(signals)
    signals.requires_grad = True

    print(signals)
    ani_gsg_features = AniGSG_model(coordinates, signals)

    loss_fun = torch.nn.MSELoss()
    loss = loss_fun(ani_gsg_features, torch.rand_like(ani_gsg_features))
    loss.backward()
    coord_grad = coordinates.grad
    nans = torch.where(torch.isnan(coord_grad)==True)[0]
    if nans.shape[0] > 0 :
        print('Coord grad has nan values')
    else:
        print("Model loaded and executed successfully")



if __name__=="__main__":

    wavelet_steps = 4
    scf_flags = (True, True, False)
    mol_idx = 117
    nans_count = test_saved_model(mol_idx)
