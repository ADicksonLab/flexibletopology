import sys
import os.path as osp
import numpy as np
import pickle as pkl

import simtk.openmm.app as omma
import simtk.openmm.openmm as omm
from simtk import unit

from flexibletopology.utils.integrators import CustomVerletInt

import warnings
warnings.filterwarnings("ignore")

INPUTS_PATH = 'inputs'

WATER_PSF = 'water_box.psf'
WATER_CRD = 'water_box.crd'

BOXLX = 20
BOXLY = 20
BOXLZ = 20

TEMP = 300 * unit.kelvin
PRESSURE = 1.0 * unit.bar
FRIC_COEFF = 1.0 / unit.picosecond
TIME_STEP = 0.002 * unit.picosecond

CENTRAL_POS = 0.5* BOXLX / 10 # in nm
WIDTH = 0.5 # nm
GHOST_MASS = 1.0 #amu

USE_CUDA = False



def getParameters(sim, n_ghosts):
    pars = sim.context.getParameters()

    par_array = np.zeros((n_ghosts,4))
    for i in range(n_ghosts):
        tmp_lambda = pars[f'lambda_g{i}']
        tmp_charge = pars[f'charge_g{i}']
        tmp_sigma = pars[f'sigma_g{i}']
        tmp_epsilon = pars[f'epsilon_g{i}']
        par_array[i] = np.array([tmp_lambda,
                                 tmp_charge,
                                 tmp_sigma,
                                 tmp_epsilon])

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

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: n_ghosts")
        sys.exit()
    else:
        n_ghosts = int(sys.argv[1])

    # load in psf and add ghost particles

    psf = omma.CharmmPsfFile(osp.join(INPUTS_PATH,
                                      WATER_PSF))
    psf.setBox(BOXLX * unit.angstroms,
               BOXLY * unit.angstroms,
               BOXLZ * unit.angstroms)

    crd = omma.CharmmCrdFile(osp.join(INPUTS_PATH,
                                      WATER_CRD))
    n_part_system = len(crd.positions)

    ghost_positions = np.random.normal(loc=CENTRAL_POS,
                                       scale=WIDTH,
                                       size=(n_ghosts, 3))

    ghost_charges = np.random.normal(loc=0.0,
                                       scale=0.2,
                                       size=(n_ghosts))

    crd.positions.extend(unit.quantity.Quantity(ghost_positions,
                                                unit.nanometers))

    params = read_params(osp.join(INPUTS_PATH, 'toppar.str'))

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

        gs_force.addGlobalParameter(f'charge_g{i}', ghost_charges[i]) # default value
        gs_force.addGlobalParameter(f'sigma_g{i}', 0.1) # default value, nm
        gs_force.addGlobalParameter(f'epsilon_g{i}', 1.0) # default value, kJ/mol
        gs_force.addGlobalParameter(f'lambda_g{i}', 1.0) # default value

        gs_force.addEnergyParameterDerivative(f'charge_g{i}')
        gs_force.addEnergyParameterDerivative(f'sigma_g{i}')
        gs_force.addEnergyParameterDerivative(f'epsilon_g{i}')
        gs_force.addEnergyParameterDerivative(f'lambda_g{i}')

        for j in range(n_part_system):
            gs_force.addParticle([sys_charge[j], sys_sigma[j], sys_epsilon[j]])

        # for each force term you need to add ALL the particles even though we only use one of them!
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
    simulation.context.setPositions(crd.positions)

    print('Before min: E=', simulation.context.getState(getEnergy=True).getPotentialEnergy())
    getEnergyComponents(simulation, n_ghosts)

    simulation.minimizeEnergy(tolerance=100.0*unit.kilojoule/unit.mole, maxIterations=1000)
    print('After min: E=', simulation.context.getState(getEnergy=True).getPotentialEnergy())
    getEnergyComponents(simulation, n_ghosts)

    # write out positions and attributes
    pos = simulation.context.getState(getPositions=True).getPositions()
    pos_arr = np.array(pos.value_in_unit(unit.nanometer))
    np.savetxt(f'{INPUTS_PATH}/wbox_pos{n_ghosts}.npy', pos_arr)

    par = getParameters(simulation, n_ghosts)
    np.savetxt(f'{INPUTS_PATH}/wbox_par{n_ghosts}.npy', par)
