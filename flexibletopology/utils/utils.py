import sys
import os
import os.path as osp
import numpy as np
import pickle as pkl

import torch
from torch.autograd import Variable

import torchani
from flexibletopology.mlmodels.gsg import GSG
from flexibletopology.mlmodels.ani_gsg import AniGSG


def save_gsg_model(max_wavelet_scale=4,
                   radial_cutoff=0.52,
                   sm_operators=(True, True, True),
                   platform='cpu',
                   save_path='gsg.pt'):

    GSG_model = GSG(max_wavelet_scale=max_wavelet_scale,
                    radial_cutoff=radial_cutoff,
                    sm_operators=sm_operators)

    device = torch.device(platform)
    GSG_model.to(device)
    GSG_model.double()

    script_module = torch.jit.script(GSG_model)

    try:
        script_module = torch.jit.script(GSG_model)
        script_module.save(save_path)
        print("The model saved successfully")
    except:

        print("Can not save the model")


def save_anigsg_model(max_wavelet_scale=4,
                      radial_cutoff=0.52,
                      sm_operators=(True, True, True),
                      platform='cpu',
                      save_path='anigsg.pt'):

    base_dir = os.path.dirname(os.path.realpath(__file__))
    ani_params_file = '../resources/ani_params/ani-1ccx_8x_nm.params'

    consts_file = os.path.join(base_dir, ani_params_file)

    AniGSG_model = AniGSG(max_wavelet_scale=max_wavelet_scale,
                          radial_cutoff=radial_cutoff,
                          sm_operators=sm_operators,
                          consts_file=consts_file)

    device = torch.device(platform)
    AniGSG_model.to(device)
    AniGSG_model.double()

    try:
        script_module = torch.jit.script(AniGSG_model)
        script_module.save(save_path)
        print("The model saved successfully")
    except:

        print("Can not save the model")
