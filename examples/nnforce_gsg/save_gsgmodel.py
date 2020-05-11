import numpy as np

import pickle as pkl
import torch
from torch.autograd import Variable
from flexibletopology.mlmodels.GSGraph import GSGraph
import warnings

warnings.filterwarnings("ignore")

MODEL_SAVE_PATH = 'inputs/gsg_model.pt'
DATASET = 'openchem_3D_8_110.pkl'


if __name__=="__main__":

    #load the data
    with open(f'inputs/{DATASET}', 'rb') as pklf:
        data = pkl.load(pklf)

    #number of atoms 27
    mol_idx = 26
    #get the coordinates
    coords = np.copy(data['coords'][mol_idx])
    signals = np.copy(data['gaff_signals_notype'][mol_idx])
    features = np.copy(data['gaff_features_notype'][mol_idx])

    #set the GSG parameters
    wavelet_num_steps = 8
    radial_cutoff = 0.75
    scf_flags= (True, True, False)

    coords = torch.from_numpy(coords)
    coords.requires_grad = True
    signals = torch.from_numpy(signals)
    signals.requires_grad = True

    #construct the Torch GSG model
    model = GSGraph(wavelet_num_steps=wavelet_num_steps,
                    radial_cutoff=radial_cutoff,
                    scf_flags=scf_flags)
    device = torch.device('cpu')
    model.to(device)
    model.double()



    try:
        traced_script_module = torch.jit.trace(model, (coords, signals))
        traced_script_module.save(MODEL_SAVE_PATH)
        print("The model saved successfuly")
    except:

       print("Can not save the model")
