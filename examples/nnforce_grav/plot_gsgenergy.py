import  sys
import os.path as osp
import numpy as np
import h5py
import matplotlib.pyplot as plt

H5_TRAJFILE_PATH = 'outputs/gravforces_traj.h5'

if not osp.exists(H5_TRAJFILE_PATH):
    print(f"Can not find file {H5_TRAJFILE_PATH}")
    sys.exit()

with h5py.File(H5_TRAJFILE_PATH, 'r') as h5:
    time = np.array(h5['time'])
    nn_energies = np.array(h5['nn_potentialEnergy'])

plt.plot(time, nn_energies)
plt.xlabel('time (ps)')
plt.ylabel('Grav energy (kcal/mol)')
plt.show()
