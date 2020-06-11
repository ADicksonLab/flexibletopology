import os
import os.path as osp
import numpy as np
import pickle as pkl

import torch
from torch.autograd import Variable

import torchani
from flexibletopology.mlmodels.AniGSGraph import AniGSGraph

import warnings

warnings.filterwarnings("ignore")


MODEL_SAVE_PATH = 'inputs/ani_gsg_model.pt'
DATASET_NAME = 'inputs/openchem_3D_8_110.pkl'


if __name__=="__main__":

    #read molecule properties from the file
    with open(DATASET_NAME, 'rb') as pklf:
        data = pkl.load(pklf)

    #number of atoms 8
    mole_idx = 117

    #get the coordinates (nm)
    coordinates = np.copy(data['coords'][mole_idx]) / 10
    signals = np.copy(data['gaff_signals_notype'][mole_idx])


    #set the GSG parameters
    wavelet_num_steps = 4
    #nm
    radial_cutoff = 0.75
    scf_flags = (True, True, False)

    AniGSG_model = AniGSGraph(wavelet_num_steps=wavelet_num_steps,
                    radial_cutoff=radial_cutoff,
                    scf_flags=scf_flags)

    coordinates = torch.from_numpy(coordinates)
    coordinates.requires_grad = True

    signals = torch.from_numpy(signals)
    signals.requires_grad = True


    device = torch.device('cpu')
    AniGSG_model.to(device)
    AniGSG_model.double()

    traced_script_module = torch.jit.trace(AniGSG_model,
                                           (coordinates, signals))

    try:
        traced_script_module = torch.jit.trace(AniGSG_model,
                                           (coordinates, signals))
        traced_script_module.save(MODEL_SAVE_PATH)
        print("The model saved successfully")
    except:

       print("Can not save the model")
