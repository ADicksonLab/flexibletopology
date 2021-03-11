import os.path as osp
import sys
import numpy as np
import pickle as pkl
import simtk.openmm.app as omma
import simtk.openmm.openmm as omm
from simtk import unit

from flexibletopology.utils.integrators import CustomVerletInt

import warnings
warnings.filterwarnings("ignore")



INPUTS_PATH = './inputs'

WATER_PSF = osp.join(INPUTS_PATH, 'water_box.psf')
WATER_CRD = osp.join(INPUTS_PATH, 'water_box.crd')

BOXLX = 20
BOXLY = 20
BOXLZ = 20

TEMP = 300 * unit.kelvin
PRESSURE = 1.0 * unit.bar
FRIC_COEFF = 1.0 / unit.picosecond
TIME_STEP = 0.002 * unit.picosecond

CENTRAL_POS = 0.5* BOXLX / 10 # in nm
WIDTH = 0.05 # nm
GHOST_MASS = 1.0 #amu

USE_CUDA = False

MLFORCESCALE = 1.0

def getParameters(sim, n_ghosts):
    pars = sim.context.getParameters()

    par_array = np.zeros((n_ghosts,4))
    for i in range(n_ghosts):
        tmp_lambda = pars[f'lambda_g{i}']
        tmp_charge = pars[f'charge_g{i}']
        tmp_sigma = pars[f'sigma_g{i}']
        tmp_epsilon = pars[f'epsilon_g{i}']
        par_array[i] = np.array([tmp_lambda, tmp_charge,
                                 tmp_sigma, tmp_epsilon])

    return par_array

def getEnergyComponents(sim, n_ghosts):
    """Gets the energy components from all of the auxiliary forces"""

    # ghost-system energies
    for i in range(n_ghosts):
        gs_state = sim.context.getState(getEnergy=True,
                                         groups={7+i})
        gs_energy = gs_state.getPotentialEnergy() / unit.kilocalorie_per_mole
        print(f'GS{i}-system, energy = {gs_energy: .7e}')


def read_params(filename):
    extlist = ['rtf', 'prm', 'str']

    parFiles = ()
    for line in open(filename, 'r'):
        if '!' in line: line = line.split('!')[0]
        parfile = line.strip()
        if len(parfile) != 0:
            ext = parfile.lower().split('.')[-1]
            if not ext in extlist:
                continue
            parFiles += (osp.join(INPUTS_PATH, parfile), )

    params = omma.CharmmParameterSet(*parFiles)
    return params



def create_mlforce(ghost_particle_idxs):
    start_idx = 1227
    mlmodel_name = 'gsg_model.pt'
    data_file_name = f'T{start_idx}_W4_110_gsg.pkl'


    dataset_path = osp.join(INPUTS_PATH, data_file_name)

    with open(dataset_path, 'rb') as pklf:
        data = pkl.load(pklf)


    target_features = data['target_features']

    ex_mlforce = mlforce.PyTorchForce(file=osp.join(INPUTS_PATH, mlmodel_name),
                                      targetFeatures=target_features,
                                      particleIndices=ghost_particle_idxs,
                                      scale=MLFORCESCALE)
    return ex_mlforce


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: position_file parameter_file")
        sys.exit()
    else:
        pos_file = sys.argv[1]
        par_file = sys.argv[2]

    # load in psf and add ghost particles

    psf = omma.CharmmPsfFile(WATER_PSF)
    psf.setBox(BOXLX * unit.angstroms,
               BOXLY * unit.angstroms,
               BOXLZ * unit.angstroms)

    crd = omma.CharmmCrdFile(WATER_CRD)
    n_part_system = len(crd.positions)

    sys_positions = np.loadtxt(pos_file)   # in nm
    g_parameters = np.loadtxt(par_file)  # lambda, charge, sigma, epsilon

    n_ghosts = g_parameters.shape[0]

    params = read_params(osp.join(INPUTS_PATH,'toppar.str'))

    system = psf.createSystem(params,
                              nonbondedMethod=omma.forcefield.CutoffPeriodic,
                              nonbondedCutoff=1*unit.nanometers,
                              constraints=omma.forcefield.HBonds)

    psf_ghost_chain = psf.topology.addChain(id='G')
    psf_ghost_res = psf.topology.addResidue('ghosts',
                                            psf_ghost_chain)

    for i in range(n_ghosts):
        system.addParticle(GHOST_MASS)
        psf.topology.addAtom('G{0}'.format(i),
                             omma.Element.getBySymbol('Ar'),
                             psf_ghost_res,
                             'G{0}'.format(i))

    # Needs to have the same number of total particles as the system, so
    # add the ghost particles to the nbforce with zero interaction
    forces = {force.__class__.__name__ : force for force in system.getForces()}
    nb_force = forces['NonbondedForce']
    for i in range(n_ghosts):
        nb_force.addParticle(0.0, #charge
                             0.1, #sigma (nm)
                             0.0) #epsilon (kJ/mol)

    # build the lists of particle parameters for fixed topology (system) atoms
    sys_sigma = []
    sys_epsilon = []
    sys_charge = []
    for i in range(n_part_system):
        tmp = nb_force.getParticleParameters(i)
        sys_charge.append(tmp[0].value_in_unit(unit.elementary_charge))
        sys_sigma.append(tmp[1].value_in_unit(unit.nanometer))
        sys_epsilon.append(tmp[2].value_in_unit(unit.kilojoule_per_mole))

    # add one custom force for each ghost particle
    # Note: Coulomb constant k = 138.9417 kJ/mol * nm/e^2
    gs_force_idxs = []


    for i in range(n_ghosts):
        energy_function = f'4*lambda_g{i}*epsilon*((sigma/r)^12-(sigma/r)^6)'
        energy_function += f'+ 138.9417*lambda_g{i}*charge1*charge_g{i}/r;'
        energy_function += f'sigma = 0.5*(sigma1+sigma_g{i}); epsilon = sqrt(epsilon1*epsilon_g{i})'
        gs_force = omm.CustomNonbondedForce(energy_function)
        gs_force.addPerParticleParameter('charge')
        gs_force.addPerParticleParameter('sigma')
        gs_force.addPerParticleParameter('epsilon')

        gs_force.addGlobalParameter(f'lambda_g{i}', g_parameters[i, 0]) # default value
        gs_force.addGlobalParameter(f'charge_g{i}', g_parameters[i, 1]) # default value
        gs_force.addGlobalParameter(f'sigma_g{i}', g_parameters[i, 2]) # default value, nm
        gs_force.addGlobalParameter(f'epsilon_g{i}', g_parameters[i, 3]) # default value, kJ/mol

        gs_force.addEnergyParameterDerivative(f'charge_g{i}')
        gs_force.addEnergyParameterDerivative(f'sigma_g{i}')
        gs_force.addEnergyParameterDerivative(f'epsilon_g{i}')
        gs_force.addEnergyParameterDerivative(f'lambda_g{i}')

        for j in range(n_part_system):
            gs_force.addParticle([sys_charge[j],
                                  sys_sigma[j],
                                  sys_epsilon[j]])

        # for each force term you need to add ALL the particles even
        # though we only use one of them!
        for j in range(n_ghosts):
            gs_force.addParticle([0.0, 0.1, 0.0])

        gs_force.addInteractionGroup(set(range(n_part_system)),
                                     set([n_part_system + i]))

        # set the NonbondedMethod to use a cutoff (default
        gs_force.setNonbondedMethod(gs_force.CutoffPeriodic)
        gs_force.setCutoffDistance(1.0) #nm

        # add the force to the system
        gs_force_idxs.append(system.addForce(gs_force))
        gs_force.setForceGroup(7+i)


    # Set platform
    if USE_CUDA:
        platform = omm.Platform.getPlatformByName('CUDA')
        prop = dict(CudaPrecision='single')
    else:
        prop = {}
        platform = omm.Platform.getPlatformByName('Reference')


    integrator = CustomVerletInt(n_ghosts, timestep=TIME_STEP)
    simulation = omma.Simulation(psf.topology, system, integrator, platform, prop)
    simulation.context.setPositions(sys_positions)

    n_iter = 10
    for iteration in range(n_iter):
        # Set alchemical state
        # Run some dynamics
        integrator.step(2)
        for i in range(n_ghosts):
            lambda_g = simulation.context.getParameter(f'lambda_g{i}')
            charge_g = simulation.context.getParameter(f'charge_g{i}')
            gsforce_state = simulation.context.getState(getEnergy=True, groups={7+i})
            gs_energy = gsforce_state.getPotentialEnergy() / unit.kilocalorie_per_mole
            print(f'Energy index {i}, energy = {gs_energy: .7e},'\
                  f'lambda_g = {lambda_g: .3f}, charge_g = {charge_g: .3f}')
