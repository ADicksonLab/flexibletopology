import numpy as np

import pickle as pkl
import torch
from torch.autograd import Variable
from flexibletopology.mlmodels.GSGraph import GSGraph

MODEL_SVAE_PATH = 'inputs/gsg_model.pt'

if __name__=="__main__":

    FEATURES_FLAG = (True, True, True)


    with open('inputs/openchem_3D4_110.pkl', 'rb') as pklf:
        data = pkl.load(pklf)

    #number of atoms 27
    idx_start = 26
    idx_end = 28


    #get the coordinates
    coords = np.copy(data['coords'][idx_start])/10

    #Define an array of signals, shape: (num_atoms, 3)[cahrge, radius,
    #epsilon]

    target_features = np.copy(data['gaff_features_notype'][idx_end])
    signals = np.copy(data['gaff_signals_notype'][idx_end])
    tmp = signals[2]
    signals[2] = signals[6]
    signals[6] = tmp


    #set the GSG parameters
    wavelet_num_steps = 4
    radial_cutoff = 7.5


    coords = torch.from_numpy(coords)
    coords.requires_grad = True
    signals = torch.from_numpy(signals)

    model = GSGraph(features=(True, True, False))
    device = torch.device('cpu')
    model.to(device)
    model.double()

    loss_fn = torch.nn.MSELoss()

    aa = model(coords, signals)

    traced_script_module = torch.jit.trace(model, (coords, signals))
    traced_script_module.save(MODEL_SVAE_PATH)

    #saved model test with different with molecules of different
    #number of atoms
    module = torch.jit.load(MODEL_SVAE_PATH)
    coords2 = Variable(torch.rand(20, 3).double())
    coords2.requires_grad = True
    signals2 = Variable(torch.rand(20, 6).double())
    out = module(coords2, signals2)
    loss = loss_fn(out, torch.rand(120, 1).double())
    loss.backward()
    print(coords2.grad)
    print(out.shape)
