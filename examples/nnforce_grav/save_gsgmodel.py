import numpy as np

import torch
from torch.autograd import Variable

from flexibletopology.mlmodels.grav import GravPotential

import warnings

warnings.filterwarnings("ignore")

MODEL_SAVE_PATH = 'inputs/grav_model.pt'
NPART = 20
NSIG = 10

if __name__=="__main__":

    #construct the Torch GSG model
    model = GravPotential(forceConstant=200, #kcal/mol
                          radius=0.5) #nm 
    device = torch.device('cpu')
    model.to(device)
    model.double()

    # generate random coordinates and signals
    coords = np.random.random((NPART,3))*2.0
    signals = np.random.random((NPART,NSIG))

    coords = torch.from_numpy(coords)
    coords.requires_grad = True
    signals = torch.from_numpy(signals)
    signals.requires_grad = True

    try:
        traced_script_module = torch.jit.trace(model, (coords, signals))
        traced_script_module.save(MODEL_SAVE_PATH)
        print("The model saved successfuly")
    except:

       print("Can not save the model")
