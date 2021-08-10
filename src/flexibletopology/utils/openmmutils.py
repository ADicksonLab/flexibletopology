import os.path as osp
import numpy as np
from simtk import unit
import simtk.openmm.app as omma


def read_params(filename, parfiles_path):
    extlist = ['rtf', 'prm', 'str']

    parFiles = ()
    for line in open(osp.join(parfiles_path, filename), 'r'):
        if '!' in line:
            line = line.split('!')[0]
        parfile = line.strip()
        if len(parfile) != 0:
            ext = parfile.lower().split('.')[-1]
            if not ext in extlist:
                continue
            parFiles += (osp.join(parfiles_path, parfile), )

    params = omma.CharmmParameterSet(*parFiles)
    return params


def getParameters(sim, n_ghosts):
    pars = sim.context.getParameters()

    par_array = np.zeros((n_ghosts, 4))
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


def getForceByClass(system, klass):
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, klass):
            return f
