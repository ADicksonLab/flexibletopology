import os
import os.path as osp
import numpy as np
import pickle as pkl

import torch
from torch.autograd import Variable

import torchani
from flexibletopology.mlmodels.AniGSGraph import AniGSGraph

MODEL_SAVE_PATH = 'inputs/ani_gsg_model.pt'
DATASET_NAME = 'inputs/openchem_3D_8_110.pkl'
TORCHANI_PATH = os.path.dirname(osp.realpath(torchani.__file__))
TORCHANI_PARAMS_FILE = '../torchani/resources/ani-1ccx_8x/rHCNO-5.2R_16-3.5A_a4-8.params'

if __name__=="__main__":

    #read molecule properties from the file
    with open('inputs/openchem_3D4_110.pkl', 'rb') as pklf:
        data = pkl.load(pklf)

    #number of atoms 27
    idx_molecule = 26
    idx_end = 28

    #get the coordinates
    coordinates = np.copy(data['coords'][idx_molecule])
    signals = np.copy(data['gaff_signals_notype'][idx_molecule])
    target_coordinates = np.copy(data['coords'][idx_end])
    target_signals = np.copy(data['gaff_signals_notype'][idx_end])
    target_coordinates = torch.from_numpy(target_coordinates)
    target_coordinates.requires_grad = True
    target_signals = torch.from_numpy(target_signals)

    coordinates = torch.from_numpy(coordinates)
    coordinates.requires_grad = True

    signals = torch.from_numpy(signals)
    signals.requires_grad = True
    num_atoms = coordinates.shape[0]

    # Consider all atoms as carbon C=6
    atom_types = ''
    for i in range(num_atoms):
        atom_types += 'C'


    #create signals from TorchANI model
    const_file = osp.join(TORCHANI_PATH, TORCHANI_PARAMS_FILE)
    consts = torchani.neurochem.Constants(const_file)
    aev_computer = torchani.AEVComputer(**consts)

    species = consts.species_to_tensor(atom_types).unsqueeze(0)
    _, target_aev_signals = aev_computer((species, target_coordinates.unsqueeze(0)))

    ani_gsg = AniGSGraph(8, 7.5, (True, True, False))

    features = ani_gsg(coordinates, signals)
    output = torch.randn_like(features, dtype=features.dtype)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(features, features)
    import ipdb; ipdb.set_trace()

    # aa = model(coords, signals)

    # traced_script_module = torch.jit.trace(model, (coords, signals))
    # traced_script_module.save(MODEL_SVAE_PATH)

    # #saved model test with different with molecules of different
    # #number of atoms
    # module = torch.jit.load(MODEL_SVAE_PATH)
    # coords2 = Variable(torch.rand(20, 3).double())
    # coords2.requires_grad = True
    # signals2 = Variable(torch.rand(20, 6).double())
    # out = module(coords2, signals2)
    # loss = loss_fn(out, torch.rand(120, 1).double())
    # loss.backward()
    # print(coords2.grad)
    # print(out.shape)
