import sys
import os
import os.path as osp
import pickle as pkl
import numpy as np

import torch
from flexibletopology.mlmodels.AniGSGraph import AniGSGraph

INPUTS_PATH = 'inputs'
DATASET_NAME = 'openchem_3D_8_110.pkl'


SCF_FLAGS = (True, True, False)
WAVELET_NUM_STEPS = 4
RADIAL_CUTOFF = 0.52

IDX_START = 117
IDX_END = 177

def save_data(dataset_name, idx_start, idx_end):

    dataset_path = osp.join(INPUTS_PATH, DATASET_NAME)

    with open(dataset_path, 'rb') as pklf:
        data = pkl.load(pklf)

    initial_signals = np.copy(data['gaff_signals_notype'][idx_start])
    initial_coords = np.copy(data['coords'][idx_start]) / 10

    target_signals = np.copy(data['gaff_signals_notype'][idx_end])
    target_coords = np.copy(data['coords'][idx_end]) / 10

    #run ANIGSG to get target features


    AniGSG_model = AniGSGraph(wavelet_num_steps=WAVELET_NUM_STEPS,
                              radial_cutoff=RADIAL_CUTOFF,
                              scf_flags=SCF_FLAGS)


    target_coords = torch.from_numpy(target_coords)
    target_coords.requires_grad = True

    target_signals = torch.from_numpy(target_signals)
    target_signals.requires_grad = True

    target_features = AniGSG_model(target_coords, target_signals)

    data = {'initial_coords': initial_coords,
            'initial_signals': initial_signals,
            'target_features': target_features.detach().numpy()}


    #Save data into a pickle file
    scf_flgs_str = ''.join([str(int(flag)) for flag in SCF_FLAGS])
    outfile_name = f'S{start_idx}_E{idx_end}_W{WAVELET_NUM_STEPS}_{scf_flgs_str}.pkl'
    save_path = os.path.join(INPUTS_PATH, outfile_name)
    with open(save_path, 'wb') as wfile:
        pkl.dump(data, wfile)


if __name__=='__main__':

    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("arguments: start_idx, end_idx")
        exit()
    else:
        start_idx = int(sys.argv[1])
        end_idx = int(sys.argv[2])
        save_data(DATASET_NAME, start_idx, end_idx)
