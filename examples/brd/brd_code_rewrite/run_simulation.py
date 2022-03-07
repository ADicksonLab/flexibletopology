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

import openmm.app as omma
import openmm as omm
import simtk.unit as unit
from sys import stdout
import time

from flexibletopology.utils.integrators import CustomHybridIntegratorConstCharge
from flexibletopology.utils.reporters import H5Reporter
from flexibletopology.utils.openmmutils import read_params
import mlforce
import sys

from system_build_util import BRDSystemBuild

import warnings
warnings.filterwarnings("ignore")

# get the GPU number
import socket
print(f'hostname: {socket.gethostname()}')

# import openmm from source
omm.Platform.loadPluginsFromDirectory(
    '/pathto/your/pkgs/openmm/lib/plugins')

# Give arguments
if sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("arguments: run_num")
else:
    run_num = int(sys.argv[1])
    timestep = float(sys.argv[2])
    
# set random seed
seed_num = np.random.randint(0, high=10000)
print(seed_num)
np.random.seed(seed_num)

# set input paths
SIM_INPUTS_PATH = f'./simulation_inputs/run{run_num}/'
INPUTS_PATH = './inputs/'
TARGET_IDX = 124
MODEL_PATH = osp.join(INPUTS_PATH, 'ani_model_cpu.pt')
TARGET_PDB = osp.join(INPUTS_PATH, f'target{TARGET_IDX}.pdb')
TARGET_FILE = osp.join(INPUTS_PATH,f'T{TARGET_IDX}_ani.pkl')
TOPPAR_STR = ('toppar.str')

SYSTEM_FILE = osp.join(SIM_INPUTS_PATH, f'run{run_num}_system.pkl')
TOP_FILE = osp.join(SIM_INPUTS_PATH, f'run{run_num}_topology.pkl')
POS_FILE = osp.join(SIM_INPUTS_PATH, f'run{run_num}_positions.pkl')
PAR_FILE = osp.join(SIM_INPUTS_PATH, f'run{run_num}_parameters.pkl')
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
TIMESTEP = timestep*unit.picoseconds
NUM_STEPS = 10000
REPORT_STEPS = 50 
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

    with open(COEFFS_FILE, "rb") as f:
        coeffs = pkl.load(f)

    with open(BOUNDS_FILE, "rb") as f:
        bounds = pkl.load(f)


    #_______________CREATE SIM OBJECT_______________#
    print('Creating simulation object')
    
    system.addForce(omm.MonteCarloBarostat(PRESSURE, TEMPERATURE))

    print('System done')
    n_ghosts = len(parameters)
    integrator = CustomHybridIntegratorConstCharge(n_ghosts, TEMPERATURE, FRICTION_COEFFICIENT,
                                    TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=bounds)
    simulation = omma.Simulation(psf_top, system, integrator, platform, prop)

    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    
    print('Updating the signal values')
    # update the signal values
    for i in range(n_ghosts):
        simulation.context.setParameter(f"charge_g{i}", parameters[i][0])
        simulation.context.setParameter(f"sigma_g{i}", parameters[i][1])
        simulation.context.setParameter(f"epsilon_g{i}", parameters[i][2])
        simulation.context.setParameter(f"lambda_g{i}", parameters[i][3])        

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
