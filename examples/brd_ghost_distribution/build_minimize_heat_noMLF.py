"""
This script builds, equilibrates, and heats a system, without adding MLForce.
"""
import os
import os.path as osp
import pickle as pkl
import numpy as np
import mdtraj as mdj

import openmm.app as omma
import openmm as omm
import openmm.unit as unit
from sys import stdout
import time
import sys

from flexibletopology.utils.integrators import CustomHybridIntegratorRestrictedChargeVariance

from flexibletopology.utils.reporters import H5Reporter
from flexibletopology.utils.openmmutils import read_params, getParameters, setParameters
import sys

from contforceplugin import ContForce
from system_build_util import SystemBuild
import warnings
warnings.filterwarnings("ignore")

# get the GPU number
import socket
print(f'hostname: {socket.gethostname()}')

# Give arguments
if sys.argv[1] == "-h" or sys.argv[1] == "--help" or len(sys.argv) < 5:
    print("arguments: n_ghosts run_num openmm_dir output_base_folder")
    exit()
else:
    n_ghosts = int(sys.argv[1])
    run_num = int(sys.argv[2])
    openmm_path = sys.argv[3]
    BASE_FOLDER = sys.argv[4]

plugins_path = os.path.join(openmm_path,'lib','plugins')

if not os.path.exists(openmm_path):
    print(f"Error! openmm_path: {openmm_path} does not exist")
elif not os.path.exists(plugins_path):
    print(f"Error! openmm_path: {openmm_path} does not contain lib/plugins directory")
    print("e.g.: /path/to/your/conda/pkgs/openmm-7.7.0-py39h9717219_0/lib/plugins")

# import openmm from source
omm.Platform.loadPluginsFromDirectory(plugins_path)

# set input paths
#PLATFORM = 'Reference'
PLATFORM = 'CUDA'
INPUTS_PATH = './inputs/'

#### System specific information
SYSTEM_PSF = osp.join(INPUTS_PATH, 'brd2_water_trim.psf') # name of psf file 
SYSTEM_PDB = osp.join(INPUTS_PATH, 'brd_water_trim.pdb')  # name of pdb file
BS_SELECTION_STRING = "resid 88 29 81 24" # selection of residues where the centroid is the middle of the binding site
SYSTEM_CONT_FORCE_IDXS = [1380,1381,1382] # These are indices of atoms in the target that will be made to be continuous with the FT atoms
                                           # (set to empty list if not using)

sigma_LowerBound = 0.2
sigma_coeff = 100000.0
rest_coeff = 100000.0
    
# set random seed
seed_num = np.random.randint(0, high=10000)    
print("Random seed:",seed_num)
np.random.seed(seed_num)
    
TOPPAR_STR = ('toppar.str')

# set output paths
OUTPUTS_PATH = osp.join(BASE_FOLDER,f'build_outputs/',f'g{n_ghosts}',f'run{run_num}')
SIM_TRAJ = osp.join(OUTPUTS_PATH, f'traj.dcd')

# load files
psf = omma.CharmmPsfFile(SYSTEM_PSF)
crd = omma.PDBFile(SYSTEM_PDB)
pdb_file = mdj.load_pdb(SYSTEM_PDB)
pos_arr = np.array(crd.positions.value_in_unit(unit.nanometers))

# MD simulations settings
#TEMPERATURES = [10, 20, 50, 100, 150, 200, 250, 300]
TEMPERATURES = [10, 20, 50]
FRICTION_COEFFICIENT = 1/unit.picosecond
TIMESTEP = 0.002*unit.picoseconds
NUM_STEPS = [10000 for t in TEMPERATURES]
#NUM_STEPS[-1] = 1000000

GHOST_MASS = 12 # AMU
REPORT_STEPS = 2000

# system building values
WIDTH = 0.3 # nm
# MIN_DIST = 0.04 # nm
MIN_DIST = 0.05 # nm

# force group values
ghostghost_group = 29
systemghost_group = 31

# minimization values
TOL = 100
MAXITR = 1000

# barostat values
PRESSURE = 1 * unit.bar

# bounds on the signals  
BOUNDS = {'charge': (-0.90, 0.90),
        'sigma': (sigma_LowerBound, 0.5), 
        'epsilon': (0.03, 1.50),
        'lambda': (1.0, 1.0)}


# Set up platform information
if PLATFORM == 'CUDA':
    print("Using CUDA platform..")
    platform = omm.Platform.getPlatformByName('CUDA')
    prop = dict(CudaPrecision='single')

else:
    print("Using Reference platform..")
    prop = {}
    platform = omm.Platform.getPlatformByName('Reference')

# build the system, minimize, and heat
if __name__ == '__main__':


    #_______________BUILD SYSTEM & SIMULATION OBJECT_______________#
    
    bs_idxs = pdb_file.topology.select(BS_SELECTION_STRING)

    print("n_ghosts is ",n_ghosts)
    n_system = pdb_file.n_atoms
    gst_idxs = list(range(n_system, n_system+n_ghosts))
    
    con_force = ContForce()
    con_force.addBond(gst_idxs, len(gst_idxs), 0.18, 10000)
    
    if len(SYSTEM_CONT_FORCE_IDXS) > 0:
        cont_force_idxs = gst_idxs + SYSTEM_CONT_FORCE_IDXS
        con_force.addBond(cont_force_idxs, len(cont_force_idxs), 0.25, 10000)

    BUILD_UTILS = SystemBuild(psf=psf, crd=crd, pdb=pdb_file, n_ghosts=n_ghosts,
                              toppar_str=TOPPAR_STR, inputs_path=INPUTS_PATH,
                              width=WIDTH, binding_site_idxs=bs_idxs,
                              min_dist=MIN_DIST, 
                              gg_group=ghostghost_group, sg_group=systemghost_group,
                              ghost_mass=GHOST_MASS, attr_bounds=BOUNDS,
                              contForce=con_force)
    
    print('Building the system..')
    system, initial_signals, n_ghosts, psf_top, crd_pos, _ = BUILD_UTILS.build_system_forces()

    tmp = crd_pos[gst_idxs[0]]
    crd_pos[gst_idxs[0]] = crd_pos[gst_idxs[-1]]
    crd_pos[gst_idxs[-1]] = tmp
    
    print('System built')
        
    coeffs = {'lambda': rest_coeff,
              'charge': rest_coeff,
              'sigma': sigma_coeff,
              'epsilon': rest_coeff}

    # build the hybrid integrator and the simulation object
   
    integrator = CustomHybridIntegratorRestrictedChargeVariance(n_ghosts, TEMPERATURES[0], FRICTION_COEFFICIENT,
                                                TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation = omma.Simulation(psf_top, system, integrator, platform, prop)

    simulation.context.setPositions(crd_pos)

    # add reporters                                                                     
    if not osp.exists(OUTPUTS_PATH):
        os.makedirs(OUTPUTS_PATH)

    pre_min_positions = simulation.context.getState(getPositions=True).getPositions()
    omma.PDBFile.writeFile(psf.topology, pre_min_positions, open(osp.join(OUTPUTS_PATH,'struct_before_min.pdb'), 'w'))

    #_______________MINIMIZE_______________#
    print('Running minimization')
    print('Before min: E=', simulation.context.getState(getEnergy=True).getPotentialEnergy())
    begin = time.time()
    simulation.minimizeEnergy(tolerance=TOL ,maxIterations=MAXITR)
    end = time.time()
    print('After min: E=', simulation.context.getState(getEnergy=True).getPotentialEnergy())

    # save a PDB of the minimized positions
    latest_state = simulation.context.getState(getPositions=True)
    latest_par = getParameters(simulation, n_ghosts)

    omma.PDBFile.writeFile(psf_top, latest_state.getPositions(), open(osp.join(OUTPUTS_PATH,f'minimized_pos.pdb'),'w'))

    print("Minimization Ends")
    print(f"Minimization run time = {np.round(end - begin, 3)}s")

    print('Heating system')
    begin = time.time()

    for temp_idx, TEMP in enumerate(TEMPERATURES):
        H5REPORTER_FILE = osp.join(OUTPUTS_PATH,f'traj{temp_idx}.h5')
        # integrator = CustomHybridIntegratorConstCharge(n_ghosts, TEMP*unit.kelvin, FRICTION_COEFFICIENT,
        #                                                TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

        integrator = CustomHybridIntegratorRestrictedChargeVariance(n_ghosts, TEMP*unit.kelvin, FRICTION_COEFFICIENT,
                                                       TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)
        

        # _______________HEAT SYSTEM_______________ #
       
        

        simulation = omma.Simulation(psf_top, system, integrator, platform, prop)

        simulation.context.setState(latest_state)
        setParameters(simulation, latest_par)
        
        simulation.reporters.append(H5Reporter(H5REPORTER_FILE,
                                               reportInterval=REPORT_STEPS,
                                               coordinates=False,
                                               velocities=False,
                                               assignments=False,
                                               global_variables=True,
                                               global_variable_forces=True,
                                               groups=systemghost_group, num_ghosts=n_ghosts))

        
        simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH,
                                                                        f'heating{temp_idx}.dcd'),
                                                              REPORT_STEPS))

        simulation.reporters.append(omma.StateDataReporter(osp.join(OUTPUTS_PATH,f'heat_SDR_out{temp_idx}.csv'),
                                                           REPORT_STEPS,
                                                           step=True,
                                                           potentialEnergy=True,
                                                           temperature=True))

        simulation.step(NUM_STEPS[temp_idx])

        latest_par = getParameters(simulation, n_ghosts)    
        latest_state = simulation.context.getState(getPositions=True)

    end = time.time()
    print(f"Heating run time = {np.round(end - begin, 3)}s")
    print('Done heating system; dcd saved')



    
    #_______________SAVE SIM INPUTS_______________#
    
    # save the system, topology, simulation object, positions, and parameters
    print(" n_ghosts",  n_ghosts)
    print('Saving simulation input files')

    with open(osp.join(OUTPUTS_PATH, 'system.pkl'), 'wb') as new_file:
        pkl.dump(system, new_file)
 
    with open(osp.join(OUTPUTS_PATH, 'topology.pkl'), 'wb') as new_file:
        pkl.dump(psf_top, new_file)

    final_pos = simulation.context.getState(getPositions=True).getPositions()

    with open(osp.join(OUTPUTS_PATH, 'positions.pkl'), 'wb') as new_file:
        pkl.dump(final_pos, new_file)
    print(" n_ghosts",  n_ghosts)

    par = getParameters(simulation, n_ghosts)
    print("parameters: ", par)
    
    with open(osp.join(OUTPUTS_PATH, 'parameters.pkl'), 'wb') as new_file:
        pkl.dump(par, new_file)

    with open(osp.join(OUTPUTS_PATH, 'coeffs.pkl'), 'wb') as new_file:
        pkl.dump(coeffs, new_file)

    with open(osp.join(OUTPUTS_PATH, 'bounds.pkl'), 'wb') as new_file:
        pkl.dump(BOUNDS, new_file)


