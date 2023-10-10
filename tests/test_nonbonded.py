"""
This is an integration test that ensures flexible topology
nonbonded interactions matches those modelled by OpenMM.
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
from flexibletopology.utils.openmmutils import read_params, nb_params_from_charmm_psf, add_ghosts_to_system
from flexibletopology.forces.nonbonded import add_gs_force, add_ghosts_to_nb_forces

import warnings
warnings.filterwarnings("ignore")

TOL = 1e-3  # test tolerance for energies
BIGTOL = 1  # test tolerance for total energies with long range corrections
BIGTOL_FORCE = 10  # test tolerance for total energies with long range corrections

def removeBondedForces(system):
    has_extra_forces = True
    ok_force_list = ['NonbondedForce','CustomNonbondedForce']
    while has_extra_forces:
        has_extra_forces = False
        for i,f in enumerate(system.getForces()):
            if f.getName() not in ok_force_list:
                system.removeForce(i)
                has_extra_forces = True
                break

    return system

# set input paths
#PLATFORM = 'Reference'
PLATFORM = 'CUDA'
INPUTS_PATH = './inputs/'

#### System specific information
PSF = {}
PSF_elec = {}
PSF_LJ = {}
PDB = {}
PSF['two_water'] = osp.join(INPUTS_PATH, 'two_waters.psf') # name of psf file
PSF['one_water'] = osp.join(INPUTS_PATH, 'one_water.psf') # name of psf file
PSF_LJ['two_water'] = osp.join(INPUTS_PATH, 'two_waters_lj.psf') # name of psf file
PSF_LJ['one_water'] = osp.join(INPUTS_PATH, 'one_water_lj.psf') # name of psf file 

PDB['two_water'] = osp.join(INPUTS_PATH, 'two_waters.pdb')  # name of pdb file
BOX_SIZE = 100.0 * unit.nanometers
sigma_LowerBound = 0.2
BOUNDS = {'charge': (-0.90, 0.90),
        'sigma': (sigma_LowerBound, 0.5), 
        'epsilon': (0.03, 1.50),
        'lambda': (1.0, 1.0)}

sigma_coeff = 200000.0
rest_coeff = 200000.0
coeffs = {'lambda': rest_coeff,
          'charge': rest_coeff,
          'sigma': sigma_coeff,
          'epsilon': rest_coeff}

FRICTION_COEFFICIENT = 1/unit.picosecond
TIMESTEP = 0.002*unit.picoseconds

TOPPAR_STR = ('toppar.str')
params = read_params(TOPPAR_STR, INPUTS_PATH)

GHOST_MASS = 12 # AMU

# Set up platform information
if PLATFORM == 'CUDA':
    print("Using CUDA platform..")
    platform = omm.Platform.getPlatformByName('CUDA')
    prop = dict(CudaPrecision='single')

else:
    print("Using Reference platform..")
    prop = {}
    platform = omm.Platform.getPlatformByName('Reference')

def testNonbonded():
    print("Testing total nonbonded energies:")    
    # step 1: build the two water system without flexible topology; get the nonbonded energies and forces

    tag = 'two_water'
    psf1 = omma.CharmmPsfFile(PSF[tag])
    crd = omma.PDBFile(PDB[tag])

    psf1.setBox(BOX_SIZE,
                BOX_SIZE,
                BOX_SIZE)

    system1 = psf1.createSystem(params,
                                nonbondedMethod=omma.forcefield.CutoffPeriodic,
                                nonbondedCutoff=1*unit.nanometers,
                                constraints=omma.forcefield.HBonds)

    system1 = removeBondedForces(system1)

    integrator1 = CustomHybridIntegratorRestrictedChargeVariance(0, 300, FRICTION_COEFFICIENT,
                                                                 TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation1 = omma.Simulation(psf1.topology, system1, integrator1, platform, prop)
    simulation1.context.setPositions(crd.positions)

    omm_state = simulation1.context.getState(getEnergy=True,getForces=True)
    omm_energy = omm_state.getPotentialEnergy()
    omm_forces = omm_state.getForces()

    # step 2: build the one water system, with the second water represented by flexible topology; get energies, forces; compare

    tag = 'one_water'

    psf2 = omma.CharmmPsfFile(PSF[tag])

    psf2.setBox(BOX_SIZE,
                BOX_SIZE,
                BOX_SIZE)

    system2 = psf2.createSystem(params,
                                nonbondedMethod=omma.forcefield.CutoffPeriodic,
                                nonbondedCutoff=1*unit.nanometers,
                                constraints=omma.forcefield.HBonds)

    sys_nb_params = nb_params_from_charmm_psf(psf2)

    n_ghosts = 3
    n_part_system = 3
    sg_group = 31
    system2, new_psf = add_ghosts_to_system(system2, psf2, n_ghosts, GHOST_MASS)
    system2, exclusion_list = add_ghosts_to_nb_forces(system2, n_ghosts, n_part_system)

    initial_attr=sys_nb_params
    initial_attr['lambda'] = [1 for i in range(n_ghosts)]

    system2 = add_gs_force(system2, n_ghosts=n_ghosts, n_part_system=n_part_system, initial_attr=initial_attr, group_num=sg_group,
                           sys_attr=sys_nb_params, nb_exclusion_list=exclusion_list)

    system2 = removeBondedForces(system2)
    for i,f in enumerate(system2.getForces()):
        f.setForceGroup(i)

        integrator2 = CustomHybridIntegratorRestrictedChargeVariance(0, 300, FRICTION_COEFFICIENT,
                                                                     TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation2 = omma.Simulation(new_psf.topology, system2, integrator2, platform, prop)
    simulation2.context.setPositions(crd.positions)

    gh_state = simulation2.context.getState(getEnergy=True,getForces=True)
    gh_energy = gh_state.getPotentialEnergy()
    gh_forces = gh_state.getForces()

    assert np.abs(omm_energy.value_in_unit(unit.kilojoule/unit.mole) - gh_energy.value_in_unit(unit.kilojoule/unit.mole)) < BIGTOL
    diff = np.array(omm_forces.value_in_unit(unit.kilojoule/(unit.nanometer*unit.mole))) - np.array(gh_forces.value_in_unit(unit.kilojoule/(unit.nanometer*unit.mole)))

    assert np.abs(diff).max() < BIGTOL_FORCE

    print("omm_energy: ",omm_energy)
    print("gh_energy: ",gh_energy)

    print("omm_forces: ", omm_forces)
    print("gh_forces: ", gh_forces)

def testVLJ_only():
    
    print("Testing LJ energies:")
    # step 1: build the two water system without flexible topology; get the nonbonded energies and forces

    tag = 'two_water'
    psf1 = omma.CharmmPsfFile(PSF[tag])
    crd = omma.PDBFile(PDB[tag])
    
    psf1.setBox(BOX_SIZE,
                BOX_SIZE,
                BOX_SIZE)

    system1 = psf1.createSystem(params,
                                nonbondedMethod=omma.forcefield.CutoffPeriodic,
                                nonbondedCutoff=1*unit.nanometers,
                                constraints=omma.forcefield.HBonds)

    sys_nb_params = nb_params_from_charmm_psf(psf1)
    
    system1 = removeBondedForces(system1)
    f = system1.getForce(0)
    for i in [0,1,2]:
        for j in [3,4,5]:
            f.addException(i,j,0.0,
                           0.5*(sys_nb_params['sigma'][i] + sys_nb_params['sigma'][j]),
                           np.sqrt(sys_nb_params['epsilon'][i]* sys_nb_params['epsilon'][j]))

    integrator1 = CustomHybridIntegratorRestrictedChargeVariance(0, 300, FRICTION_COEFFICIENT,
                                                                 TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation1 = omma.Simulation(psf1.topology, system1, integrator1, platform, prop)
    simulation1.context.setPositions(crd.positions)

    omm_state = simulation1.context.getState(getEnergy=True,getForces=True)
    omm_energy = omm_state.getPotentialEnergy()
    omm_forces = omm_state.getForces()

    # step 2: build the one water system, with the second water represented by flexible topology; get energies, forces; compare

    tag = 'one_water'

    psf2 = omma.CharmmPsfFile(PSF[tag])

    psf2.setBox(BOX_SIZE,
                BOX_SIZE,
                BOX_SIZE)

    system2 = psf2.createSystem(params,
                                nonbondedMethod=omma.forcefield.CutoffPeriodic,
                                nonbondedCutoff=1*unit.nanometers,
                                constraints=omma.forcefield.HBonds)

    sys_nb_params = nb_params_from_charmm_psf(psf2)

    n_ghosts = 3
    n_part_system = 3
    sg_group = 31
    system2, new_psf = add_ghosts_to_system(system2, psf2, n_ghosts, GHOST_MASS)
    system2, exclusion_list = add_ghosts_to_nb_forces(system2, n_ghosts, n_part_system)

    initial_attr=sys_nb_params
    initial_attr['lambda'] = [1 for i in range(n_ghosts)]
    initial_attr['charge'] = [0 for i in range(n_ghosts)]
    
    system2 = add_gs_force(system2, n_ghosts=n_ghosts, n_part_system=n_part_system, initial_attr=initial_attr, group_num=sg_group,
                           sys_attr=sys_nb_params, nb_exclusion_list=exclusion_list)

    system2 = removeBondedForces(system2)
    for i,f in enumerate(system2.getForces()):
        f.setForceGroup(i)

        integrator2 = CustomHybridIntegratorRestrictedChargeVariance(0, 300, FRICTION_COEFFICIENT,
                                                                     TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation2 = omma.Simulation(new_psf.topology, system2, integrator2, platform, prop)
    simulation2.context.setPositions(crd.positions)

    gh_state = simulation2.context.getState(getEnergy=True,getForces=True)
    gh_energy = gh_state.getPotentialEnergy()
    gh_forces = gh_state.getForces()

    assert np.abs(omm_energy.value_in_unit(unit.kilojoule/unit.mole) - gh_energy.value_in_unit(unit.kilojoule/unit.mole)) < TOL
    diff = np.array(omm_forces.value_in_unit(unit.kilojoule/(unit.nanometer*unit.mole))) - np.array(gh_forces.value_in_unit(unit.kilojoule/(unit.nanometer*unit.mole)))

    assert np.abs(diff).max() < TOL

    print("omm_energy: ",omm_energy)
    print("gh_energy: ",gh_energy)

    print("omm_forces: ", omm_forces)
    print("gh_forces: ", gh_forces)

def testVelec_only():
    
    # step 1: build the two water system without flexible topology; get the nonbonded energies and forces
    print("Testing electrostatic energies:")
    
    tag = 'two_water'
    psf1 = omma.CharmmPsfFile(PSF[tag])
    crd = omma.PDBFile(PDB[tag])
    
    psf1.setBox(BOX_SIZE,
                BOX_SIZE,
                BOX_SIZE)

    system1 = psf1.createSystem(params,
                                nonbondedMethod=omma.forcefield.CutoffPeriodic,
                                nonbondedCutoff=1*unit.nanometers,
                                constraints=omma.forcefield.HBonds)

    sys_nb_params = nb_params_from_charmm_psf(psf1)
    
    system1 = removeBondedForces(system1)
    f = system1.getForce(0)
    for i in [0,1,2]:
        for j in [3,4,5]:
            f.addException(i,j,sys_nb_params['charge'][i]*sys_nb_params['charge'][j],
                           0.5*(sys_nb_params['sigma'][i] + sys_nb_params['sigma'][j]),
                           0.0)

    integrator1 = CustomHybridIntegratorRestrictedChargeVariance(0, 300, FRICTION_COEFFICIENT,
                                                                 TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation1 = omma.Simulation(psf1.topology, system1, integrator1, platform, prop)
    simulation1.context.setPositions(crd.positions)

    omm_state = simulation1.context.getState(getEnergy=True,getForces=True)
    omm_energy = omm_state.getPotentialEnergy()
    omm_forces = omm_state.getForces()

    # step 2: build the one water system, with the second water represented by flexible topology; get energies, forces; compare

    tag = 'one_water'

    psf2 = omma.CharmmPsfFile(PSF[tag])

    psf2.setBox(BOX_SIZE,
                BOX_SIZE,
                BOX_SIZE)

    system2 = psf2.createSystem(params,
                                nonbondedMethod=omma.forcefield.CutoffPeriodic,
                                nonbondedCutoff=1*unit.nanometers,
                                constraints=omma.forcefield.HBonds)

    sys_nb_params = nb_params_from_charmm_psf(psf2)

    n_ghosts = 3
    n_part_system = 3
    sg_group = 31
    system2, new_psf = add_ghosts_to_system(system2, psf2, n_ghosts, GHOST_MASS)
    system2, exclusion_list = add_ghosts_to_nb_forces(system2, n_ghosts, n_part_system)

    initial_attr=sys_nb_params
    initial_attr['lambda'] = [1 for i in range(n_ghosts)]
    initial_attr['epsilon'] = [0 for i in range(n_ghosts)]
    
    system2 = add_gs_force(system2, n_ghosts=n_ghosts, n_part_system=n_part_system, initial_attr=initial_attr, group_num=sg_group,
                           sys_attr=sys_nb_params, nb_exclusion_list=exclusion_list)

    system2 = removeBondedForces(system2)
    for i,f in enumerate(system2.getForces()):
        f.setForceGroup(i)

        integrator2 = CustomHybridIntegratorRestrictedChargeVariance(0, 300, FRICTION_COEFFICIENT,
                                                                     TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation2 = omma.Simulation(new_psf.topology, system2, integrator2, platform, prop)
    simulation2.context.setPositions(crd.positions)

    gh_state = simulation2.context.getState(getEnergy=True,getForces=True)
    gh_energy = gh_state.getPotentialEnergy()
    gh_forces = gh_state.getForces()

    assert np.abs(omm_energy.value_in_unit(unit.kilojoule/unit.mole) - gh_energy.value_in_unit(unit.kilojoule/unit.mole)) < TOL
    diff = np.array(omm_forces.value_in_unit(unit.kilojoule/(unit.nanometer*unit.mole))) - np.array(gh_forces.value_in_unit(unit.kilojoule/(unit.nanometer*unit.mole)))

    assert np.abs(diff).max() < TOL

    print("omm_energy: ",omm_energy)
    print("gh_energy: ",gh_energy)

    print("omm_forces: ", omm_forces)
    print("gh_forces: ", gh_forces)

def testVLJ_only_psf():
    print("Testing total nonbonded energies:")    
    # step 1: build the two water system without flexible topology; get the nonbonded energies and forces

    tag = 'two_water'
    psf1 = omma.CharmmPsfFile(PSF_LJ[tag])
    crd = omma.PDBFile(PDB[tag])

    psf1.setBox(BOX_SIZE,
                BOX_SIZE,
                BOX_SIZE)

    system1 = psf1.createSystem(params,
                                nonbondedMethod=omma.forcefield.CutoffPeriodic,
                                nonbondedCutoff=1*unit.nanometers,
                                constraints=omma.forcefield.HBonds)

    system1 = removeBondedForces(system1)

    integrator1 = CustomHybridIntegratorRestrictedChargeVariance(0, 300, FRICTION_COEFFICIENT,
                                                                 TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation1 = omma.Simulation(psf1.topology, system1, integrator1, platform, prop)
    simulation1.context.setPositions(crd.positions)

    omm_state = simulation1.context.getState(getEnergy=True,getForces=True)
    omm_energy = omm_state.getPotentialEnergy()
    omm_forces = omm_state.getForces()

    # step 2: build the one water system, with the second water represented by flexible topology; get energies, forces; compare

    tag = 'one_water'

    psf2 = omma.CharmmPsfFile(PSF_LJ[tag])

    psf2.setBox(BOX_SIZE,
                BOX_SIZE,
                BOX_SIZE)

    system2 = psf2.createSystem(params,
                                nonbondedMethod=omma.forcefield.CutoffPeriodic,
                                nonbondedCutoff=1*unit.nanometers,
                                constraints=omma.forcefield.HBonds)

    sys_nb_params = nb_params_from_charmm_psf(psf2)

    n_ghosts = 3
    n_part_system = 3
    sg_group = 31
    system2, new_psf = add_ghosts_to_system(system2, psf2, n_ghosts, GHOST_MASS)
    system2, exclusion_list = add_ghosts_to_nb_forces(system2, n_ghosts, n_part_system)

    initial_attr=sys_nb_params
    initial_attr['lambda'] = [1 for i in range(n_ghosts)]

    system2 = add_gs_force(system2, n_ghosts=n_ghosts, n_part_system=n_part_system, initial_attr=initial_attr, group_num=sg_group,
                           sys_attr=sys_nb_params, nb_exclusion_list=exclusion_list)

    system2 = removeBondedForces(system2)
    for i,f in enumerate(system2.getForces()):
        f.setForceGroup(i)

        integrator2 = CustomHybridIntegratorRestrictedChargeVariance(0, 300, FRICTION_COEFFICIENT,
                                                                     TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation2 = omma.Simulation(new_psf.topology, system2, integrator2, platform, prop)
    simulation2.context.setPositions(crd.positions)

    gh_state = simulation2.context.getState(getEnergy=True,getForces=True)
    gh_energy = gh_state.getPotentialEnergy()
    gh_forces = gh_state.getForces()

    assert np.abs(omm_energy.value_in_unit(unit.kilojoule/unit.mole) - gh_energy.value_in_unit(unit.kilojoule/unit.mole)) < TOL
    diff = np.array(omm_forces.value_in_unit(unit.kilojoule/(unit.nanometer*unit.mole))) - np.array(gh_forces.value_in_unit(unit.kilojoule/(unit.nanometer*unit.mole)))

    assert np.abs(diff).max() < TOL

    print("omm_energy: ",omm_energy)
    print("gh_energy: ",gh_energy)

    print("omm_forces: ", omm_forces)
    print("gh_forces: ", gh_forces)


def testNonbonded_dist():
    # step 1: build the two water system without flexible topology; get the nonbonded energies and forces

    dist_fracs = np.linspace(0.5,1.0,20)
    dists = []
    omm_energies = []
    gh_energies = []
    
    tag = 'two_water'
    psf1 = omma.CharmmPsfFile(PSF[tag])
    crd = omma.PDBFile(PDB[tag])

    psf1.setBox(BOX_SIZE,
                BOX_SIZE,
                BOX_SIZE)

    system1 = psf1.createSystem(params,
                                nonbondedMethod=omma.forcefield.CutoffPeriodic,
                                nonbondedCutoff=1*unit.nanometers,
                                constraints=omma.forcefield.HBonds)

    system1 = removeBondedForces(system1)

    integrator1 = CustomHybridIntegratorRestrictedChargeVariance(0, 300, FRICTION_COEFFICIENT,
                                                                 TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation1 = omma.Simulation(psf1.topology, system1, integrator1, platform, prop)

    # move the 5th atom closer to the 0th atom
    orig_pos5 = np.array(crd.positions[5].value_in_unit(unit.nanometers))
    orig_pos0 = np.array(crd.positions[0].value_in_unit(unit.nanometers))

    for frac in dist_fracs:
        new_pos5 = orig_pos5 + (1-frac)*(orig_pos0-orig_pos5)
        d = np.sqrt(np.sum(np.square(new_pos5 - orig_pos0)))
        dists.append(d)
        crd.positions[5] = unit.Quantity(omm.Vec3(*new_pos5),unit.nanometer)
    
        simulation1.context.setPositions(crd.positions)

        omm_state = simulation1.context.getState(getEnergy=True,getForces=True)
        omm_energies.append(omm_state.getPotentialEnergy())

    # step 2: build the one water system, with the second water represented by flexible topology; get energies, forces; compare

    tag = 'one_water'

    psf2 = omma.CharmmPsfFile(PSF[tag])

    psf2.setBox(BOX_SIZE,
                BOX_SIZE,
                BOX_SIZE)

    system2 = psf2.createSystem(params,
                                nonbondedMethod=omma.forcefield.CutoffPeriodic,
                                nonbondedCutoff=1*unit.nanometers,
                                constraints=omma.forcefield.HBonds)

    sys_nb_params = nb_params_from_charmm_psf(psf2)

    n_ghosts = 3
    n_part_system = 3
    sg_group = 31
    system2, new_psf = add_ghosts_to_system(system2, psf2, n_ghosts, GHOST_MASS)
    system2, exclusion_list = add_ghosts_to_nb_forces(system2, n_ghosts, n_part_system)

    initial_attr=sys_nb_params
    initial_attr['lambda'] = [1 for i in range(n_ghosts)]

    system2 = add_gs_force(system2, n_ghosts=n_ghosts, n_part_system=n_part_system, initial_attr=initial_attr, group_num=sg_group,
                           sys_attr=sys_nb_params, nb_exclusion_list=exclusion_list)

    system2 = removeBondedForces(system2)
    for i,f in enumerate(system2.getForces()):
        f.setForceGroup(i)

        integrator2 = CustomHybridIntegratorRestrictedChargeVariance(0, 300, FRICTION_COEFFICIENT,
                                                                     TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation2 = omma.Simulation(new_psf.topology, system2, integrator2, platform, prop)

    for frac in dist_fracs:
        new_pos5 = orig_pos5 + (1-frac)*(orig_pos0-orig_pos5)
        d = np.sqrt(np.sum(np.square(new_pos5 - orig_pos0)))
        dists.append(d)
        crd.positions[5] = unit.Quantity(omm.Vec3(*new_pos5),unit.nanometer)
    
        simulation2.context.setPositions(crd.positions)

        gh_state = simulation2.context.getState(getEnergy=True,getForces=True)
        gh_energies.append(gh_state.getPotentialEnergy())

    assert np.array(gh_energies).argmin() == np.array(omm_energies).argmin()
    # print energy vs dist
    for i in range(len(dist_fracs)):
        print(dists[i],omm_energies[i],gh_energies[i])

if __name__ == "__main__":
    testVLJ_only()
    testVelec_only()
    testVLJ_only_psf()
    testNonbonded_dist()
    testNonbonded()
    
    print("PASSED all tests")
