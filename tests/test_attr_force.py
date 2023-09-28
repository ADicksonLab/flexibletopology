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
from flexibletopology.utils.openmmutils import read_params, nb_params_from_charmm_psf, add_ghosts_to_system, getParameters, setParameters
from flexibletopology.forces.nonbonded import add_gs_force, add_ghosts_to_nb_forces

import sys

import warnings
warnings.filterwarnings("ignore")

TOL = 1e-3  # test tolerance for energies

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
TIMESTEP = 0.0*unit.picoseconds

TOPPAR_STR = ('toppar.str')
params = read_params(TOPPAR_STR, INPUTS_PATH)

GHOST_MASS = 12 # AMU

def testAttrForce(platform, prop):
    print("Testing total nonbonded energies:")    
    # step 1: build the two water system without flexible topology; get the nonbonded energies and forces

    psf = omma.CharmmPsfFile(PSF['one_water'])
    crd = omma.PDBFile(PDB['two_water'])

    psf.setBox(BOX_SIZE,
               BOX_SIZE,
               BOX_SIZE)

    n_ghosts = 3
    n_part_system = 3
    sg_group = 31
    
    # get attr forces for normal system
    system = psf.createSystem(params,
                              nonbondedMethod=omma.forcefield.CutoffPeriodic,
                              nonbondedCutoff=1*unit.nanometers,
                              constraints=omma.forcefield.HBonds)
    sys_nb_params = nb_params_from_charmm_psf(psf)
    initial_attr=sys_nb_params
    initial_attr['lambda'] = [1 for i in range(n_ghosts)]

    system, new_psf = add_ghosts_to_system(system, psf, n_ghosts, GHOST_MASS)
    system, exclusion_list = add_ghosts_to_nb_forces(system, n_ghosts, n_part_system)
    
    system = add_gs_force(system, n_ghosts=n_ghosts, n_part_system=n_part_system, initial_attr=initial_attr, group_num=sg_group,
                          sys_attr=sys_nb_params, nb_exclusion_list=exclusion_list)

    integrator = CustomHybridIntegratorRestrictedChargeVariance(3, 0, FRICTION_COEFFICIENT,
                                                                TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation = omma.Simulation(new_psf.topology, system, integrator, platform, prop)

    simulation.context.setPositions(crd.positions)

    par_derivs = simulation.context.getState(getParameterDerivatives=True).getEnergyParameterDerivatives()
    print("Par derivs:")
    for attr in ['charge','epsilon','sigma','lambda']:
        for i in range(n_ghosts):
            k = f'{attr}_g{i}'
            print(k,par_derivs[k])
    
    simulation.step(1)
    pars = simulation.context.getParameters()

    print("Attr forces:")
    for attr in ['charge','epsilon','sigma','lambda']:
        for i in range(n_ghosts):
            k = f'f{attr}_g{i}'
            k2 = f'{attr}_g{i}'
            print(k,pars[k])
            assert pars[k] + par_derivs[k2] < TOL


    # get attr forces of system again to show reproducability
    system = psf.createSystem(params,
                              nonbondedMethod=omma.forcefield.CutoffPeriodic,
                              nonbondedCutoff=1*unit.nanometers,
                              constraints=omma.forcefield.HBonds)
    sys_nb_params = nb_params_from_charmm_psf(psf)
    initial_attr=sys_nb_params
    initial_attr['lambda'] = [1 for i in range(n_ghosts)]
    
    system, new_psf = add_ghosts_to_system(system, psf, n_ghosts, GHOST_MASS)
    system, exclusion_list = add_ghosts_to_nb_forces(system, n_ghosts, n_part_system)
    
    system = add_gs_force(system, n_ghosts=n_ghosts, n_part_system=n_part_system, initial_attr=initial_attr, group_num=sg_group,
                          sys_attr=sys_nb_params, nb_exclusion_list=exclusion_list)

    integrator = CustomHybridIntegratorRestrictedChargeVariance(3, 0, FRICTION_COEFFICIENT,
                                                                TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation = omma.Simulation(new_psf.topology, system, integrator, platform, prop)

    simulation.context.setPositions(crd.positions)

    par_derivs_dup = simulation.context.getState(getParameterDerivatives=True).getEnergyParameterDerivatives()
    print("Par derivs (reprod.):")
    for attr in ['charge','epsilon','sigma','lambda']:
        for i in range(n_ghosts):
            k = f'{attr}_g{i}'
            print(k,par_derivs_dup[k])
            assert par_derivs_dup[k] - par_derivs[k] < TOL

    # swap positions of gh1 and gh2
    # get attr forces for swapped system
    system = psf.createSystem(params,
                              nonbondedMethod=omma.forcefield.CutoffPeriodic,
                              nonbondedCutoff=1*unit.nanometers,
                              constraints=omma.forcefield.HBonds)
    sys_nb_params = nb_params_from_charmm_psf(psf)
    initial_attr=sys_nb_params
    initial_attr['lambda'] = [1 for i in range(n_ghosts)]
    
    system, new_psf = add_ghosts_to_system(system, psf, n_ghosts, GHOST_MASS)
    system, exclusion_list = add_ghosts_to_nb_forces(system, n_ghosts, n_part_system)
    
    system = add_gs_force(system, n_ghosts=n_ghosts, n_part_system=n_part_system, initial_attr=initial_attr, group_num=sg_group,
                          sys_attr=sys_nb_params, nb_exclusion_list=exclusion_list)

    integrator = CustomHybridIntegratorRestrictedChargeVariance(3, 0, FRICTION_COEFFICIENT,
                                                                TIMESTEP, attr_fric_coeffs=coeffs, attr_bounds=BOUNDS)

    simulation = omma.Simulation(new_psf.topology, system, integrator, platform, prop)

    tmp = crd.positions[4]
    crd.positions[4] = crd.positions[5]
    crd.positions[5] = tmp

    simulation.context.setPositions(crd.positions)

    par_derivs_swap = simulation.context.getState(getParameterDerivatives=True).getEnergyParameterDerivatives()
    swap_idxs = [0,2,1]
    print("Par derivs (swap):")
    for attr in ['charge','epsilon','sigma','lambda']:
        for i in range(n_ghosts):
            k = f'{attr}_g{swap_idxs[i]}'
            ki = f'{attr}_g{i}'
            print(k,par_derivs_swap[k])
            assert par_derivs_swap[k] - par_derivs[ki] < TOL

    simulation.step(1)
    pars_swap = simulation.context.getParameters()

    print("Attr forces (swap):")
    for attr in ['charge','epsilon','sigma','lambda']:
        for i in range(n_ghosts):
            k = f'f{attr}_g{swap_idxs[i]}'
            ki = f'f{attr}_g{i}'
            print(k,pars_swap[k])
            assert pars_swap[k] - pars[ki] < TOL
            
def calc_attr_force_by_hand():
    crd = omma.PDBFile(PDB['two_water'])
    psf = omma.CharmmPsfFile(PSF['one_water'])

    psf.setBox(BOX_SIZE,
               BOX_SIZE,
               BOX_SIZE)
    
    system = psf.createSystem(params,
                              nonbondedMethod=omma.forcefield.CutoffPeriodic,
                              nonbondedCutoff=1*unit.nanometers,
                              constraints=omma.forcefield.HBonds)
    sys_nb_params = nb_params_from_charmm_psf(psf)

    pos = np.array(crd.positions.value_in_unit(unit.nanometer))
    f = {}
    f['lambda'] = [-calc_flambda(i,pos,sys_nb_params) for i in range(3)]
    f['sigma'] = [0,0,0]# [calc_fsigma(i,pos,sys_nb_params) for i in range(3)]
    f['charge'] = [-calc_fcharge(i,pos,sys_nb_params) for i in range(3)]
    f['epsilon'] = [0,0,0] # [calc_fepsilon(i,pos,sys_nb_params) for i in range(3)]

    print("Attr forces (by hand):")
    for attr in ['charge','epsilon','sigma','lambda']:
        for i in range(3):
            k = f'f{attr}_g{i}'
            print(k,f[attr][i])
            
def calc_flambda(i,pos,sys_nb_params):
    res = 0
    for j in range(3):
        r = np.sqrt(np.sum(np.square(pos[i+3]-pos[j])))
        epstot = np.sqrt(sys_nb_params['epsilon'][j]*sys_nb_params['epsilon'][i])
        sigma= 0.5*(sys_nb_params['sigma'][j] + sys_nb_params['sigma'][i])
        sor6 = (sigma/r)**6
        sor12 = sor6**2
        
        res += 4.0*epstot*(sor12-sor6)+138.935456*sys_nb_params['charge'][j]*sys_nb_params['charge'][i]/r
    return res

def calc_fcharge(i,pos,sys_nb_params):
    res = 0
    for j in range(3):
        r = np.sqrt(np.sum(np.square(pos[i+3]-pos[j])))
        
        res += 138.935456*sys_nb_params['charge'][j]/r
    return res

if __name__ == "__main__":

    PLATFORM = sys.argv[1]
    
    if PLATFORM == 'CUDA':
        print("Using CUDA platform..")
        platform = omm.Platform.getPlatformByName('CUDA')
        prop = dict(CudaPrecision='single')
        
    else:
        print("Using Reference platform..")
        prop = {}
        platform = omm.Platform.getPlatformByName('Reference')

    calc_attr_force_by_hand()
    testAttrForce(platform,prop)
    
    print("PASSED all tests")
