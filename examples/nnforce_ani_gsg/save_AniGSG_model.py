import os
import os.path as osp
import numpy as np
import pickle as pkl

import torch
from torch.autograd import Variable

import torchani
from flexibletopology.mlmodels.GSGraph import GSGraph


import warnings

warnings.filterwarnings("ignore")

MODEL_SAVE_PATH = 'inputs/ani_gsg_model.pt'
DATASET_NAME = 'inputs/openchem_3D_8_110.pkl'
TORCHANI_PATH = os.path.dirname(osp.realpath(torchani.__file__))
TORCHANI_PARAMS_FILE = '../torchani/resources/ani-1ccx_8x/rHCNO-5.2R_16-3.5A_a4-8.params'

if __name__=="__main__":

    #read molecule properties from the file
    with open(DATASET_NAME, 'rb') as pklf:
        data = pkl.load(pklf)

    #number of atoms 8
    mol_idx = 26

    #get the coordinates
    coordinates = torch.from_numpy(np.copy([data['coords'][mol_idx]])).to(torch.float32)
    signals = torch.from_numpy(np.copy(data['gaff_signals_notype'][mol_idx])).to(torch.float32)
    num_atoms = coordinates.shape[1]
    coordinates.requires_grad = True

    # Consider all atoms as carbon C=6
    atom_types = ''
    for i in range(num_atoms):
        atom_types+='C'


    #create signals from TorchANI model
    const_file = osp.join(TORCHANI_PATH, TORCHANI_PARAMS_FILE)
    consts = torchani.neurochem.Constants(const_file)
    aev_computer = torchani.AEVComputer(**consts)

    species = consts.species_to_tensor(atom_types).unsqueeze(0)
    _, aev_signals = aev_computer((species, coordinates))


    #set the GSG parameters
    wavelet_num_steps = 8
    radial_cutoff = 7.5
    scf_flags= (True, True, False)

    #construct the Torch GSG model
    model = GSGraph(wavelet_num_steps=wavelet_num_steps,
                    radial_cutoff=radial_cutoff,
                    scf_flags=scf_flags)
    device = torch.device('cpu')
    model.to(device)
    model.double()



    coordinates = coordinates.squeeze(0).double()
    signals = torch.cat((aev_signals.squeeze(0),signals), 1).double()
    traced_script_module = torch.jit.trace(model, (coordinates, signals))
    traced_script_module.save(MODEL_SAVE_PATH)

    print('ANI_GSG model is saved successfuly')
