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
from termcolor import colored


from flexibletopology.utils.integrators import CustomLGIntegrator
from flexibletopology.utils.reporters import H5Reporter
from flexibletopology.utils.openmmutils import read_params
import mlforce

import warnings
warnings.filterwarnings("ignore")
import sys
import socket
print(f'hostname: {socket.gethostname()}')

#Path to openmm correct version and plug-in as used in the mlforce installation
omm.Platform.loadPluginsFromDirectory(
    '/home/bosesami/anaconda3/pkgs/openmm-7.7.0-py39h9717219_0/lib/plugins')


run_num = int(sys.argv[1])

seed_num = np.random.randint(0, high=10000)
print(seed_num)
np.random.seed(seed_num)

INPUTS_PATH = './inputs'
SYSTEM_PSF = osp.join(INPUTS_PATH, 'brd2.psf')
SYSTEM_PDB = osp.join(INPUTS_PATH, 'brd_nvt.pdb')
SAVE_PATH = './inputs'

# MD simulations settings
GHOST_MASS = 50 #Needs to be checked
TEMPERATURE = 300.0 * unit.kelvin
FRICTION_COEFFICIENT = 1/unit.picosecond
TIMESTEP = 0.001*unit.picoseconds

# Set input and output files name
NUM_STEPS = 1000
REPORT_STEPS = 1
PLATFORM = 'CUDA'
WIDTH = 0.30 #in nm
TOL= 50
MAXITR = 1000

MODEL_NAME = 'ani_model_cuda.pt'

TARGET_IDX = 124
OUTPUTS_PATH = f'CUDA_outputs/T{TARGET_IDX}_mass{GHOST_MASS}/run{run_num}/'
MODEL_PATH = osp.join(INPUTS_PATH, MODEL_NAME)
DATA_FILE = f'T{TARGET_IDX}_ani.pkl'
MLFORCESCALE = 5000

PDB = f'traj{TARGET_IDX}.pdb'
SIM_TRAJ = f'traj{TARGET_IDX}.dcd'
H5REPORTER_FILE = f'traj{TARGET_IDX}.h5'
TARGET_PDB = f'target{TARGET_IDX}.pdb'

def getParameters(sim, n_ghosts):
    pars = sim.context.getParameters()
    par_array = np.zeros((n_ghosts,4))
    for i in range(n_ghosts):
        tmp_charge = pars[f'charge_g{i}']
        tmp_sigma = pars[f'sigma_g{i}']
        tmp_epsilon = pars[f'epsilon_g{i}']
        tmp_lambda = pars[f'lambda_g{i}']
        par_array[i] = np.array([tmp_charge,tmp_sigma,tmp_epsilon,tmp_lambda])
    return par_array

def init_positions(COM_BS, WIDTH, n_ghosts, min_dist):
    
    rand_positions = []

    while len(rand_positions) < 1:

        r_pos = np.random.uniform(low=-WIDTH, high=WIDTH,size=(1, 3))
        r_pos = r_pos+COM_BS
        dists = np.linalg.norm(np.concatenate(pdb_pos) - r_pos, axis=1)
        if np.all(dists > min_dist):
            rand_positions.append(r_pos)
        
    while len(rand_positions) < n_ghosts:
        
        r_pos = np.random.uniform(low=-WIDTH, high=WIDTH,size=(1, 3))
        r_pos = r_pos+COM_BS
            
        dists_pdb = np.linalg.norm(np.concatenate(pdb_pos) - r_pos, axis=1)
        dists_gho = np.linalg.norm(np.concatenate(rand_positions) - r_pos, axis=1)
        
        if np.all(dists_pdb > min_dist) and np.all(dists_gho > min_dist):
            rand_positions.append(r_pos)
    
    return np.concatenate(rand_positions)

def read_target_mol_info(data_file_name):

    dataset_path = osp.join(INPUTS_PATH,
                            data_file_name)

    with open(dataset_path, 'rb') as pklf:
        data = pkl.load(pklf)

    return data['target_coords'], data['target_signals'], data['target_features']

if __name__ == '__main__':

    start_time = time.time()

    # reading the target features from the data file
    target_pos, target_signals, target_features = read_target_mol_info(
        DATA_FILE)
    n_ghosts = target_pos.shape[0]


    #load the positions
    print("Loading pdb..")
    crd = omma.PDBFile(SYSTEM_PDB)
    n_part_system = len(crd.positions)

    print("Loading psf..")
    # load in psf and add ghost particles
    psf = omma.CharmmPsfFile(SYSTEM_PSF)
    pos_arr = np.array(crd.positions.value_in_unit(unit.nanometers))
    pdb_pos = np.array([pos_arr])

    # get the COM_BS
    # select residues PRO24, PRO28, GLY34, ASP38, and ASN82 from pdb
    # note: pdb numbering begins at 1, mdtraj begins at 0
    pdb_file = mdj.load_pdb(SYSTEM_PDB)
    idxs = pdb_file.topology.select("resid 23" "resid 27" "resid 33" "resid 37" "resid 81")
    print(idxs)
    idxs_pos = pdb_file.xyz[0][idxs]
    WIDTH = 0.30 #nm
    COM_BS = np.mean(idxs_pos, axis=0)
    
    # calculating box length
    box_lengths = pos_arr.max(axis=0) - pos_arr.min(axis=0)
    psf.setBox(box_lengths[0] * unit.nanometers,
               box_lengths[1] * unit.nanometers,
               box_lengths[2] * unit.nanometers)

    # reading FF params
    params = read_params('toppar.str', INPUTS_PATH)

    # Initialization of ghost atom positions
    #min_dist = 2**(1/6)*0.1
    min_dist =0.15
    init_pos = init_positions(COM_BS, WIDTH, n_ghosts, min_dist)
    print('Initial positions of ghosts in nm')
    print(init_pos*10)

#    import ipdb; ipdb.set_trace()
    # Extending the brd system positions to brd+GA system 
    crd.positions.extend(unit.quantity.Quantity(init_pos, 
                                            unit.nanometers))
    print("Creating system..")
    system = psf.createSystem(params,
                              nonbondedMethod=omma.forcefield.CutoffPeriodic,
                              nonbondedCutoff=1*unit.nanometers,
                              constraints=omma.forcefield.AllBonds)

    print("Adding ghosts to topology..")
    psf_ghost_chain = psf.topology.addChain(id='G')
    psf_ghost_res = psf.topology.addResidue('ghosts',
                                            psf_ghost_chain)

    # creating a list of charges, sigmas and epsilons 
    # to be used in ga_sys custom-nonbonded force later
    sys_sigma = []
    sys_epsilon = []
    sys_charge = []
    ep_convert = -0.2390057
    for atom in psf.atom_list:
        # in units of elementary charge
        sys_charge.append(atom.charge)
        # now in units of nm
        sys_sigma.append(atom.type.rmin*0.1)
        # now a positive number in kJ/mol
        sys_epsilon.append(atom.type.epsilon/ep_convert)

    # adding ghost particles to the system
    for i in range(n_ghosts):
        system.addParticle(GHOST_MASS)
        psf.topology.addAtom('G{0}'.format(i),
                             omma.Element.getBySymbol('Ar'),
                             psf_ghost_res,
                             'G{0}'.format(i))

    # bounds on the signals  
    bounds = {'charge': (-1.27, 2.194),
              'sigma': (0.05, 0.23),
              'epsilon': (0.037, 2.63),
              'lambda': (0.0, 1.0)}

    # initializing the signals
    initial_signals = np.zeros((n_ghosts, 4))
    initial_signals[:, 0] = np.random.uniform(low=-0.3, high=+0.3, size=(n_ghosts))
    initial_signals[:, 1] = np.random.uniform(low=0.05, high=0.2, size=(n_ghosts))
    initial_signals[:, 2] = np.random.uniform(low=0.1, high=1.0, size=(n_ghosts))
    initial_signals[:, 3] = 1.0
    print('Initial attributes of ghosts (charge, sigma, epsilon, lambda)')
    print(initial_signals)
    ###### FORCES (This will go to util)

    nb_forces = []
    cnb_forces = []
    for i,force in enumerate(system.getForces()):
        force.setForceGroup(i)
        if force.__class__.__name__ == 'NonbondedForce':
            nb_forces.append(force.getForceGroup())
        if force.__class__.__name__ == 'CustomNonbondedForce':
            cnb_forces.append(force.getForceGroup())

    for fidx in nb_forces:
        nb_force = system.getForce(fidx)
        for i in range(n_ghosts):
            nb_force.addParticle(0.0, #charge
                                 0.2, #sigma (nm)
                                 0.0) #epsilon (kJ/mol)
    for fidx in cnb_forces:
        cnb_force = system.getForce(fidx)

        for gh_idx in range(n_ghosts):
            cnb_force.addParticle([0.0])

        cnb_force.addInteractionGroup(set(range(n_part_system)),
                                      set(range(n_part_system)))
        cnb_force.addInteractionGroup(set(range(n_part_system,n_part_system + n_ghosts)),
                                      set(range(n_part_system,n_part_system + n_ghosts)))

        num_exclusion = cnb_force.getNumExclusions()

    exclusion_list=[]
    for i in range(num_exclusion):
        exclusion_list.append(cnb_force.getExclusionParticles(i))

    # 1. mlforce section
    mlforce_group = 30
    # indices of ghost particles in the topology
    ghost_particle_idxs = [gh_idx for gh_idx in range(n_part_system,(n_part_system+n_ghosts))]

    # force weights for "charge", "sigma", "epsilon", "lambda"
    signal_force_weights = [4000.0, 50.0, 100.0, 2000.0]
    exmlforce = mlforce.PyTorchForce(file=MODEL_PATH,
                                     targetFeatures=target_features,
                                     particleIndices=ghost_particle_idxs,
                                     signalForceWeights=signal_force_weights,
                                     scale=MLFORCESCALE)

    exmlforce.setForceGroup(mlforce_group)
    system.addForce(exmlforce)


    # 3. custom compound bond force between only the ghost atoms
    trj = mdj.load_pdb(SYSTEM_PDB)
    #anchor_idxs = idxs
    #print(anchor_idxs)
    anchor_idxs = []
    for i in range(10,12):
        anchor_idxs.append(trj.top.residue(81).atom(i).index)
    cbf = omm.CustomCentroidBondForce(2, "0.5*k*step(distance(g1,g2) - d0)*(distance(g1,g2) - d0)^2")
    cbf.addGlobalParameter('k', 1000)
    cbf.addGlobalParameter('d0', 0.5)
    anchor_grp_idx = cbf.addGroup(anchor_idxs)
    for gh_idx in range(n_ghosts):
        gh_grp_idx = cbf.addGroup([ghost_particle_idxs[gh_idx]])
        cbf.addBond([anchor_grp_idx, gh_grp_idx])

    system.addForce(cbf)

    # 4. the GS_FORCE (ghost-system non-bonded force)
    gs_force_idxs = []
    print(colored("Adding ghost-system forces to system..",'red'))
    for gh_idx in range(n_ghosts):
        energy_function = f'4*lambda_g{gh_idx}*epsilon*(sor12-sor6)+138.9417*lambda_g{gh_idx}*charge1*charge_g{gh_idx}/r;'
        energy_function += 'sor12 = sor6^2; sor6 = (sigma/r)^6;'
        energy_function += f'sigma = 0.5*(sigma1+sigma_g{gh_idx}); epsilon = sqrt(epsilon1*epsilon_g{gh_idx})'
        gs_force = omm.CustomNonbondedForce(energy_function)

        gs_force.addPerParticleParameter('charge')
        gs_force.addPerParticleParameter('sigma')
        gs_force.addPerParticleParameter('epsilon')

        # set to initial values
        gs_force.addGlobalParameter(f'charge_g{gh_idx}', initial_signals[gh_idx, 0])
        gs_force.addGlobalParameter(f'sigma_g{gh_idx}', initial_signals[gh_idx, 1])
        gs_force.addGlobalParameter(f'epsilon_g{gh_idx}', initial_signals[gh_idx, 2])
        gs_force.addGlobalParameter(f'lambda_g{gh_idx}', initial_signals[gh_idx, 3])
        gs_force.addGlobalParameter(f'assignment_g{gh_idx}', 0)

        # adding the del(signal)s [needed in the integrator]
        gs_force.addEnergyParameterDerivative(f'lambda_g{gh_idx}')
        gs_force.addEnergyParameterDerivative(f'charge_g{gh_idx}')
        gs_force.addEnergyParameterDerivative(f'sigma_g{gh_idx}')
        gs_force.addEnergyParameterDerivative(f'epsilon_g{gh_idx}')

        # adding the systems params to the force
        for p_idx in range(n_part_system):
            gs_force.addParticle(
                [sys_charge[p_idx], sys_sigma[p_idx], sys_epsilon[p_idx]])

        # for each force term you need to add ALL the particles even
        # though we only use one of them!
        for p_idx in range(n_ghosts):
            gs_force.addParticle(
                [initial_signals[p_idx, 1], initial_signals[p_idx, 2], initial_signals[p_idx, 3]])

        # interaction between ghost and system    
        gs_force.addInteractionGroup(set(range(n_part_system)),
                                     set([n_part_system + gh_idx]))


        for j in range(len(exclusion_list)):
            gs_force.addExclusion(exclusion_list[j][0], exclusion_list[j][1])

        # periodic cutoff
        gs_force.setNonbondedMethod(gs_force.CutoffPeriodic)
        # cutoff distance in nm
        gs_force.setCutoffDistance(1.0)
        # adding the force to the system
        gs_force_idxs.append(system.addForce(gs_force))

    # 6. Harmonic position restraining potential (only for minimization)
    ext_force = omm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    ext_force.addGlobalParameter("k", 50.0*unit.kilocalories_per_mole/unit.angstroms**2)
    ext_force.addPerParticleParameter("x0")
    ext_force.addPerParticleParameter("y0")
    ext_force.addPerParticleParameter("z0")
    
    for i in range(trj.top.select('name OXT')[0]):
        ext_force.addParticle(i, crd.positions[i].value_in_unit(unit.nanometers))
    #ext_force.setForceGroup(29)
    system.addForce(ext_force)

    # Set up platform
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

    coeffs = {'lambda': 10000,
              'charge': 10000,
              'sigma': 10000,
              'epsilon': 10000}

    integrator = CustomLGIntegrator(n_ghosts, TEMPERATURE, FRICTION_COEFFICIENT,
                                    TIMESTEP, coeffs=coeffs, bounds=bounds)

    simulation = omma.Simulation(psf.topology, system, integrator,
                                 platform, prop)


    print('Platform',PLATFORM)
    print('Simulation object', simulation.context)

    simulation.context.setPositions(crd.positions)
    begin = time.time()

    # add reporers
    if not osp.exists(OUTPUTS_PATH):
        os.makedirs(OUTPUTS_PATH)

    # create a pdb file from initial positions
    #omma.PDBFile.writeFile(topology, init_pos,
    #                       open(osp.join(OUTPUTS_PATH, PDB), 'w'))
    #omma.PDBFile.writeFile(topology, target_pos,
    #                       open(osp.join(OUTPUTS_PATH, TARGET_PDB), 'w'))

    simulation.reporters.append(H5Reporter(osp.join(OUTPUTS_PATH, H5REPORTER_FILE),
                                           reportInterval=REPORT_STEPS,
                                           groups=mlforce_group, num_ghosts=n_ghosts))

    simulation.reporters.append(omma.StateDataReporter(stdout, REPORT_STEPS,
                                                       step=True,
                                                       potentialEnergy=True,
                                                       temperature=True))


    simulation.reporters.append(mdj.reporters.DCDReporter(osp.join(OUTPUTS_PATH, SIM_TRAJ),
                                                          REPORT_STEPS))

    print('Before min: E=', simulation.context.getState(getEnergy=True).getPotentialEnergy())

    simulation.minimizeEnergy(tolerance=TOL ,maxIterations=MAXITR)

    print('After min: E=', simulation.context.getState(getEnergy=True).getPotentialEnergy())
    
    position_em = simulation.context.getState(getPositions=True).getPositions()
    omma.PDBFile.writeFile(psf.topology, position_em, open(f'minim{run_num}_'+str(n_ghosts)+'.pdb','w'))

    # write out positions and signals to npy files
    pos_arr = np.array(position_em.value_in_unit(unit.nanometer))
    np.savetxt(f'BRD_pos_systemFixed.npy',pos_arr)
    par = getParameters(simulation, n_ghosts)
    np.savetxt(f'BRD_par_systemFixed.npy',par)
    print(np.subtract(par, initial_signals))

    #simulation.step(NUM_STEPS)

    # apply PBC to the saved trajectory
    # pdb = mdj.load_pdb(osp.join(INPUTS_PATH, PDB))
    # traj = mdj.load_dcd(osp.join(outputs_path, SIM_TRAJ), top=topology)
    # traj = traj.center_coordinates()
    #traj.save_dcd(osp.join(outputs_path, SIM_TRAJ))

    print("Minimization Ends")
    #print(f"Simulation Steps: {NUM_STEPS}")
    end = time.time()
    print(f"Run time = {np.round(end - begin, 3)}s")
    #simulation_time = round((TIMESTEP * NUM_STEPS).value_in_unit(unit.nanoseconds),
    #                        6)
    #print(f"Simulation time: {simulation_time}ns")
