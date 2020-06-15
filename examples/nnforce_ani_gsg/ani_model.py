import torch
import numpy as np
import pickle as pkl
from flexibletopology.mlmodels.aev import AEVComputer

DATASET_NAME = 'inputs/openchem_3D_8_110.pkl'

device = torch.device('cpu')
Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
num_species = 8


#read molecule properties from the file
with open(DATASET_NAME, 'rb') as pklf:
    data = pkl.load(pklf)

#number of atoms 8
mole_idx = 117

#get the coordinates (nm)
coordinates = np.copy(data['coords'][mole_idx])
signals = np.copy(data['gaff_signals_notype'][mole_idx])

num_species = 8
coordinates = torch.from_numpy(coordinates)

aev_computer = AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
species = torch.ones((1,8), dtype=torch.long).to(device)
_, signals = aev_computer((species, coordinates.unsqueeze(0)))
print(signals)
