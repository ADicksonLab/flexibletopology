import os.path as osp
import numpy as np
from openmm import unit
import openmm.app as omma

EP_CONVERT= -0.2390057

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

    par_dict = {}
    par_dict['lambda'] = np.array([pars[f'lambda_g{i}'] for i in range(n_ghosts)])
    par_dict['charge'] = np.array([pars[f'charge_g{i}'] for i in range(n_ghosts)])
    par_dict['sigma'] = np.array([pars[f'sigma_g{i}'] for i in range(n_ghosts)])
    par_dict['epsilon'] = np.array([pars[f'epsilon_g{i}'] for i in range(n_ghosts)])
    par_dict['assignment'] = np.array([pars[f'assignment_g{i}'] for i in range(n_ghosts)])

    return par_dict

def setParameters(sim, par_dict):
    n_ghosts = len(par_dict['lambda'])
    for attr in ['lambda','charge','sigma','epsilon','assignment']:
        for i in range(n_ghosts):
            sim.context.setParameter(f'{attr}_g{i}',par_dict[attr][i])

    return

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


def writeBondEnergyString(num_ghosts):
    # write main energy expression
    energy_string = "0.5*k*("

    for gh_idx in range(1, num_ghosts+1):
        energy_string += f'step(d{gh_idx}-dmax)*(d{gh_idx}-dmax)^2'
        if gh_idx < num_ghosts:
            energy_string += ' + '
    energy_string += ');'

    # write definitions of di variables
    for gh_idx in range(1, num_ghosts+1):
        energy_string += f' d{gh_idx}=sqrt((x{gh_idx}-cx)^2 + (y{gh_idx}-cy)^2 + (z{gh_idx}-cz)^2);'

    # define cx, cy, cz
    for a in ['x', 'y', 'z']:
        energy_string += f' c{a}=('
        for gh_idx in range(1, num_ghosts+1):
            energy_string += f'{a}{gh_idx}'
            if gh_idx < num_ghosts:
                energy_string += '+'
        energy_string += f')/{num_ghosts}'
        if a != 'z':
            energy_string += ';'

    return energy_string

def nb_params_from_charmm_psf(psf):

    params = {}
    params['sigma'] = []
    params['epsilon'] = []
    params['charge'] = []

    for atom in psf.atom_list:
        params['charge'].append(atom.charge)         # in units of elementary charge

        half_rmin = atom.type.rmin*0.1         # now in units of nm
        sigma = half_rmin*2/2**(1/6)
        params['sigma'].append(sigma)
        params['epsilon'].append(atom.type.epsilon/EP_CONVERT)          # now a positive number in kJ/mol

    return params

def add_ghosts_to_system(system, psf, n_ghosts, ghost_mass):
    psf_ghost_chain = psf.topology.addChain(id='G')
    psf_ghost_res = psf.topology.addResidue('ghosts',
                                            psf_ghost_chain)

    # adding ghost particles to the system
    for i in range(n_ghosts):
        system.addParticle(ghost_mass)
        psf.topology.addAtom(f'G{i}',
                             omma.Element.getBySymbol('Ar'),
                             psf_ghost_res,
                             f'G{i}')

    return system, psf
