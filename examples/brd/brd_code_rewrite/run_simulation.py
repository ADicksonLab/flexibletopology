"""
The script takes in output files from the build/minimize/heat code,
re-adds the mlforce, and runs a flexibletopology simulation for
the BRD system.
Output is a DCD and an H5 file.
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
SIM_INPUTS_PATH = f'./simulation_inputs/run{run_num}/'
INPUTS_PATH = './inputs/'
TARGET_IDX = 124
MODEL_PATH = osp.join(INPUTS_PATH, 'ani_model_cuda.pt')
TARGET_FILE = osp.join(INPUTS_PATH,f'T{TARGET_IDX}_ani.pkl')
TOPPAR_STR = ('toppar.str')

SYSTEM_FILE = osp.join(SIM_INPUTS_PATH, f'run{run_num}_system_noMLForce.pkl')
TOP_FILE = osp.join(SIM_INPUTS_PATH, f'run{run_num}_topology.pkl')
POS_FILE = osp.join(SIM_INPUTS_PATH, f'run{run_num}_positions.pkl')
PAR_FILE = osp.join(SIM_INPUTS_PATH, f'run{run_num}_parameters.pkl')
SF_WEIGHTS = osp.join(SIM_INPUTS_PATH, f'run{run_num}_sfweights.pkl')
TAR_FEATS = osp.join(SIM_INPUTS_PATH, f'run{run_num}_targetfeats.pkl')
MLFORCESCALE_FILE = osp.join(SIM_INPUTS_PATH, f'run{run_num}_MLFORCESCALE.pkl')
COEFFS_FILE = osp.join(SIM_INPUTS_PATH, f'run{run_num}_coeffs.pkl')
BOUNDS_FILE = osp.join(SIM_INPUTS_PATH, f'run{run_num}_bounds.pkl')

# set output paths
OUTPUTS_PATH = f'./simulation_outputs/run{run_num}/'
#OUTPUTS_PATH = f'/dickson/s2/roussey1/brd_development/simulation_outputs/run{run_num}/'
SIM_TRAJ = f'traj{TARGET_IDX}.dcd'
H5REPORTER_FILE = osp.join(OUTPUTS_PATH,f'traj{TARGET_IDX}.h5')

# MD simulations settings
TEMPERATURE = 300.0 * unit.kelvin
PRESSURE = 1 * unit.bar
FRICTION_COEFFICIENT = 1/unit.picosecond
TIMESTEP = 0.001*unit.picoseconds
NUM_STEPS = 10000
REPORT_STEPS = 10 # save data every 10 simulation steps
PLATFORM = 'CUDA'

# force group values
mlforce_group = 30

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


if __name__ == '__main__':


    #_______________LOAD INPUTS_______________#
    print('Loading input files')
    
    with open(SYSTEM_FILE, "rb") as f:
        system = pkl.load(f) # this system does not have the mlforce

    with open(TOP_FILE, "rb") as f:
        psf_top = pkl.load(f)

    with open(POS_FILE, "rb") as f:
        positions = pkl.load(f)

    with open(PAR_FILE, "rb") as f:
        parameters = pkl.load(f)
        
    with open(SF_WEIGHTS, "rb") as f:
        sf_weights = pkl.load(f)

    with open(MLFORCESCALE_FILE, "rb") as f:
        mlforce_scale = pkl.load(f)

    with open(TAR_FEATS, "rb") as f:
        target_features = pkl.load(f)

    with open(COEFFS_FILE, "rb") as f:
        coeffs = pkl.load(f)

    with open(BOUNDS_FILE, "rb") as f:
        bounds = pkl.load(f)
        
    #_______________ADD BACK MLFORCE_______________#
    print('Adding MLForce')
    
    n_ghosts = len(parameters)
    n_part_system = len(positions) - n_ghosts
    ghost_particle_idxs = [gh_idx for gh_idx in range(n_part_system,(n_part_system+n_ghosts))]

    exmlforce = mlforce.PyTorchForce(file=MODEL_PATH,
                                     targetFeatures=target_features,
                                     particleIndices=ghost_particle_idxs,
                                     signalForceWeights=sf_weights,
                                     scale=mlforce_scale)

    exmlforce.setForceGroup(mlforce_group)
    system.addForce(exmlforce)

    #_______________CREATE SIM OBJECT_______________#
    print('Creating simulation object')

    system.addForce(omm.MonteCarloBarostat(PRESSURE, TEMPERATURE))
    
    integrator = CustomHybridIntegrator(n_ghosts, TEMPERATURE, FRICTION_COEFFICIENT,
                                    TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=bounds)
    simulation = omma.Simulation(psf_top, system, integrator, platform, prop)

    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    # add reporters
    if not osp.exists(OUTPUTS_PATH):
        os.makedirs(OUTPUTS_PATH)

    simulation.reporters.append(H5Reporter(H5REPORTER_FILE,
                                           reportInterval=REPORT_STEPS,
                                           groups=mlforce_group, num_ghosts=n_ghosts))

    simulation.reporters.append(omma.StateDataReporter(osp.join(OUTPUTS_PATH + 'sim_SDR_out.csv'),
                                                       REPORT_STEPS,
                                                       step=True,
                                                       potentialEnergy=True,
                                                       temperature=True))


    simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH, SIM_TRAJ),REPORT_STEPS))

    #_______________RUN SIMULATION_______________#
    print('Running simulation')
    
    begin = time.time()
    simulation.step(NUM_STEPS)    
    end = time.time()

    print(f"Simulation run time = {np.round(end - begin, 3)}s")
    print('Done simulating system; dcd/h5 saved')
