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
    radial_cutoff = 7.5
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




    traced_script_module = torch.jit.trace(model, (coords, signals))
    traced_script_module.save(MODEL_SAVE_PATH)

    #test the save model
    loaded_model = torch.jit.load(MODEL_SAVE_PATH)
    predicted_features = loaded_model(coords, signals)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(torch.tensor(features), predicted_features)
    loss.backward()
    if loss.item()==0.0:
        print("The model saved successfuly and calculates features correctly")
    else:
        print("The model saved successfuly but calculated features are not correct")
