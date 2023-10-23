import sys
import os
import os.path as osp
import numpy as np
import pickle as pkl

import torch
from torch.autograd import Variable

import torchani
from flexibletopology.mlmodels.gsg import GSG
from flexibletopology.mlmodels.ani import Ani, AniGSG


def save_gsg_model(max_wavelet_scale=4,
                   radial_cutoff=0.52,
                   sm_operators=(True, True, True),
                   platform='cpu',
                   save_path='gsg.pt',
                   sd_params=None):

    gsg_model = GSG(max_wavelet_scale=max_wavelet_scale,
                    radial_cutoff=radial_cutoff,
                    sm_operators=sm_operators,
                    sd_params=sd_params)

    device = torch.device(platform)
    gsg_model.to(device)

    try:
        script_module = torch.jit.script(gsg_model)
        script_module.save(save_path)
        print("The model saved successfully")
    except:

        print("Can not save the model")


def save_anigsg_model(max_wavelet_scale=4,
                      radial_cutoff=0.52,
                      sm_operators=(True, True, True),
                      platform='cpu',
                      save_path='anigsg.pt',
                      sd_params=None):

    base_dir = os.path.dirname(os.path.realpath(__file__))
    ani_params_file = '../resources/ani_params/ani-1ccx_8x_nm_refined.params'

    consts_file = os.path.join(base_dir, ani_params_file)

    aniGSG_model = AniGSG(max_wavelet_scale=max_wavelet_scale,
                          radial_cutoff=radial_cutoff,
                          sm_operators=sm_operators,
                          consts_file=consts_file,
                          sd_params=sd_params)

    device = torch.device(platform)
    aniGSG_model.to(device)

    try:
        script_module = torch.jit.script(aniGSG_model)
        script_module.save(save_path)
        print("The model saved successfully")
    except:

        print("Can not save the model")


def save_ani_model(platform='cpu',
                   save_path='ani_model.pt'):

    base_dir = os.path.dirname(os.path.realpath(__file__))
    ani_params_file = '../resources/ani_params/ani-1ccx_8x_nm_refined.params'

    consts_file = os.path.join(base_dir, ani_params_file)

    ani_model = Ani(platform=platform, consts_file=consts_file)

    try:
        script_module = torch.jit.script(ani_model)
        script_module = torch.jit.freeze(script_module.eval())
        script_module.save(save_path)
        print("The model saved successfully")
    except:

        print("Can not save the model")

def build_protein_restraint_force(positions,prot_idxs):

    posresPROT = CustomExternalForce('f*(dx^2+dy^2+dz^2); \
    dx=abs(x-x0); \
    dy=abs(y-y0); \
    dz=abs(z-z0);')
    posresPROT.addGlobalParameter('f',1000.)
    posresPROT.addPerParticleParameter('x0')
    posresPROT.addPerParticleParameter('y0')
    posresPROT.addPerParticleParameter('z0')

    for at_idx in prot_idxs:
        xpos  = positions[at_idx].value_in_unit(unit.nanometers)[0]
        ypos  = positions[at_idx].value_in_unit(unit.nanometers)[1]
        zpos  = positions[at_idx].value_in_unit(unit.nanometers)[2]
        posresPROT.addParticle(at_idx, [xpos, ypos, zpos])

    return posresPROT


