"""
This script builds, equilibrates, and heats a BRD system.
The outputs can be used with the run_simualtion.py script to
run a flexibletopolgoy/mlforce simulation.
"""

import sys
import os
import os.path as osp
import pickle as pkl
import numpy as np
import mdtraj as mdj
import pandas as pd

import simtk.openmm.app as omma
import openmm as omm
import simtk.unit as unit
from sys import stdout
import time

from flexibletopology.utils.integrators import CustomHybridIntegrator
from flexibletopology.utils.reporters import H5Reporter
from flexibletopology.utils.openmmutils import read_params
import mlforce
import sys

from system_build_util import BRDSystemBuild
from system_heat_util import HeatingAndEquilibration

import warnings
warnings.filterwarnings("ignore")

# get the GPU number
import socket
print(f'hostname: {socket.gethostname()}')

# import openmm from source
omm.Platform.loadPluginsFromDirectory(
    '/home/roussey1/miniconda3/pkgs/openmm-7.7.0-py39h9717219_0/lib/plugins')

# Give arguments
if sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("arguments: run_num")
else:
    run_num = int(sys.argv[1])
    
# set random seed
seed_num = np.random.randint(0, high=10000)
print(seed_num)
np.random.seed(seed_num)

# set input paths
INPUTS_PATH = './inputs/'
TARGET_IDX = 124
SYSTEM_PSF = osp.join(INPUTS_PATH, 'brd2.psf')
SYSTEM_PDB = osp.join(INPUTS_PATH, 'brd_nvt.pdb')
MODEL_PATH = osp.join(INPUTS_PATH, 'ani_model_cuda.pt')
TARGET_FILE = osp.join(INPUTS_PATH,f'T{TARGET_IDX}_ani.pkl')
TOPPAR_STR = ('toppar.str')

# set output paths
OUTPUTS_PATH = f'./build_outputs/run{run_num}/'
SAVE_PATH = f'./simulation_inputs/run{run_num}/'
SIM_TRAJ = osp.join(OUTPUTS_PATH, f'traj{TARGET_IDX}.dcd')
H5REPORTER_FILE = osp.join(OUTPUTS_PATH,f'traj{TARGET_IDX}.h5')

# load files
psf = omma.CharmmPsfFile(SYSTEM_PSF)
crd = omma.PDBFile(SYSTEM_PDB)
pdb_file = mdj.load_pdb(SYSTEM_PDB)
pos_arr = np.array(crd.positions.value_in_unit(unit.nanometers))

# MD simulations settings
TEMPERATURE = 300.0 * unit.kelvin
FRICTION_COEFFICIENT = 1/unit.picosecond
TIMESTEP = 0.001*unit.picoseconds
NUM_STEPS = 5000
REPORT_STEPS = 5 # save data every 10 simulation steps

PLATFORM = 'CUDA'
MLFORCESCALE = 25000
GHOST_MASS = 20 # AMU

# system building values
WIDTH = 0.15 # nm
MIN_DIST = 0.15 # nm
CONVERT_FAC = -0.2390057

# force group values
ghostghost_group = 29
mlforce_group = 30
systemghost_group = 31

# minimization values
TOL = 100
MAXITR = 1000

# barostat values
PRESSURE = 1 * unit.bar

# Set up platform information
if PLATFORM == 'CUDA':
    print("Using CUDA platform..")
    platform = omm.Platform.getPlatformByName('CUDA')
    prop = dict(CudaPrecision='double')

elif PLATFORM == 'OpenCL':
    print("Using OpenCL platform..")
    platform = omm.Platform.getPlatformByName('OpenCL')
    prop = dict(OpenCLPrecision='double')

else:
    print("Using Reference platform..")
    prop = {}
    platform = omm.Platform.getPlatformByName('Reference')

# build the system, minimize, and heat
if __name__ == '__main__':


    #_______________BUILD SYSTEM & SIMULATION OBJECT_______________#
    signal_force_weights = [4000.0, 50.0, 100.0, 2000.0]
    bs_idxs = pdb_file.topology.select("resid 23" "resid 27" "resid 33" "resid 37" "resid 81")

    BUILD_UTILS = BRDSystemBuild(psf=psf, crd=crd, pdb=pdb_file, target_pkl=TARGET_FILE,
                                 toppar_str=TOPPAR_STR, inputs_path=INPUTS_PATH,
                                 ani_model=MODEL_PATH, width=WIDTH, binding_site_idxs=bs_idxs,
                                 min_dist=MIN_DIST, ep_convert=CONVERT_FAC,
                                 sf_weights=signal_force_weights, gg_group=ghostghost_group,
                                 mlforce_group=mlforce_group, sg_group=systemghost_group,
                                 mlforce_scale=MLFORCESCALE, ghost_mass=GHOST_MASS)

    print('Building the system..')
    system, initial_signal, n_ghosts, psf_top, crd_pos, target_feats = BUILD_UTILS.build_system_forces()
    print('System built')
    
    # bounds on the signals  
    bounds = {'charge': (-1.27, 2.194),
              'sigma': (0.05, 0.23),
              'epsilon': (0.037, 2.63),
              'lambda': (0.0, 1.0)}
    
    coeffs = {'lambda': 10000,
              'charge': 10000,
              'sigma': 10000,
              'epsilon': 10000}

    # build the hybrid integrator and the simulation object
    integrator = CustomHybridIntegrator(n_ghosts, TEMPERATURE, FRICTION_COEFFICIENT,
                                    TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=bounds)
    simulation = omma.Simulation(psf_top, system, integrator, platform, prop)

    simulation.context.setPositions(crd_pos)

    # add reporters                                                                     
    if not osp.exists(OUTPUTS_PATH):
        os.makedirs(OUTPUTS_PATH)

    #_______________MINIMIZE_______________#
    print('Running minimization')
    print('Before min: E=', simulation.context.getState(getEnergy=True).getPotentialEnergy())
    begin = time.time()
    simulation.minimizeEnergy(tolerance=TOL ,maxIterations=MAXITR)
    end = time.time()
    print('After min: E=', simulation.context.getState(getEnergy=True).getPotentialEnergy())

    # save a PDB of the minimized positions
    position_em = simulation.context.getState(getPositions=True).getPositions()

    if not osp.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    omma.PDBFile.writeFile(psf.topology, position_em, open(osp.join(OUTPUTS_PATH+f'minimized_pos_run{run_num}_'+str(n_ghosts)+'.pdb'),'w'))

    print("Minimization Ends")
    print(f"Minimization run time = {np.round(end - begin, 3)}s")

    
    #_______________HEAT SYSTEM_______________#
    simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH+
                                                          f'heating_step_run{run_num}.dcd'),
                                                          REPORT_STEPS))

    simulation.reporters.append(omma.StateDataReporter(osp.join(OUTPUTS_PATH + 'heat_SDR_out.csv'),
                                                       REPORT_STEPS,
                                                       step=True,
                                                       potentialEnergy=True,
                                                       temperature=True))

    print('Heating system')
    begin = time.time()

    simulation.step(NUM_STEPS)    
    
    end = time.time()
    print(f"Heating run time = {np.round(end - begin, 3)}s")
    print('Done heating system; dcd saved')

    #_______________SAVE SIM INPUTS_______________#
    
    # save the system, topology, simulation object, positions, and parameters

    # the mlforce must be removed before the system can be saved
    # it will be re-added in the run_simualtion script
    system.removeForce(9)
    with open(osp.join(SAVE_PATH + f'run{run_num}_system_noMLForce.pkl'), 'wb') as new_file:
        pkl.dump(system, new_file)
    
    with open(osp.join(SAVE_PATH + f'run{run_num}_topology.pkl'), 'wb') as new_file:
        pkl.dump(psf_top, new_file)

    final_pos = simulation.context.getState(getPositions=True).getPositions()

    with open(osp.join(SAVE_PATH + f'run{run_num}_positions.pkl'), 'wb') as new_file:
        pkl.dump(final_pos, new_file)

    par = BUILD_UTILS.getParameters(simulation, n_ghosts)

    with open(osp.join(SAVE_PATH + f'run{run_num}_parameters.pkl'), 'wb') as new_file:
        pkl.dump(par, new_file)
        
    with open(osp.join(SAVE_PATH + f'run{run_num}_sfweights.pkl'), 'wb') as new_file:
        pkl.dump(signal_force_weights, new_file)

    with open(osp.join(SAVE_PATH + f'run{run_num}_targetfeats.pkl'), 'wb') as new_file:
        pkl.dump(target_feats, new_file)

    with open(osp.join(SAVE_PATH + f'run{run_num}_MLFORCESCALE.pkl'), 'wb') as new_file:
        pkl.dump(MLFORCESCALE, new_file)

    with open(osp.join(SAVE_PATH + f'run{run_num}_coeffs.pkl'), 'wb') as new_file:
        pkl.dump(coeffs, new_file)

    with open(osp.join(SAVE_PATH + f'run{run_num}_bounds.pkl'), 'wb') as new_file:
        pkl.dump(bounds, new_file)
