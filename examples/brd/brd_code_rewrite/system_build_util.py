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

from flexibletopology.utils.openmmutils import read_params
import mlforce
import sys
    
class BRDSystemBuild(object):

    """
    A class that generates a ready-to-equilibrate BRD system.
    """

    def __init__(self, psf=None, crd=None, pdb=None, target_pkl=None, toppar_str=None, inputs_path=None,
                 ani_model=None, width=None, binding_site_idxs=None, min_dist=None, ep_convert=None,
                 sf_weights=None, gg_group=None, mlforce_group=None, sg_group=None, mlforce_scale=None,
                 ghost_mass=None):

        self.psf = psf
        self.crd = crd
        self.pdb = pdb
        self.target_pkl = target_pkl
        self.toppar_str = toppar_str
        self.inputs_path = inputs_path
        self.ani_model = ani_model
        self.width = width
        self.binding_site_idxs = binding_site_idxs
        self.ep_convert = ep_convert
        self.sf_weights = sf_weights
        self.gg_group = gg_group
        self.mlforce_group = mlforce_group
        self.sg_group = sg_group
        self.mlforce_scale = mlforce_scale
        self.ghost_mass = ghost_mass
        
        assert binding_site_idxs is not None, "Must give binding site indices"

        if min_dist is None:
            self.min_dist = 0.15 #nm
        else:
            self.min_dist = min_dist

    def getParameters(self, sim, n_ghosts):

        pars = sim.context.getParameters()
        par_array = np.zeros((n_ghosts,4))
        for i in range(n_ghosts):
            tmp_charge = pars[f'charge_g{i}']
            tmp_sigma = pars[f'sigma_g{i}']
            tmp_epsilon = pars[f'epsilon_g{i}']
            tmp_lambda = pars[f'lambda_g{i}']
            par_array[i] = np.array([tmp_charge,tmp_sigma,tmp_epsilon,tmp_lambda])

        return par_array

    def init_positions(self, COM_BS, WIDTH, n_ghosts, min_dist, pdb_pos):

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


    def read_target_mol_info(self, mol_file_name):

        with open(mol_file_name, 'rb') as pklf:
            data = pkl.load(pklf)

        return data['target_coords'], data['target_signals'], data['target_features']


    def generate_COM(self, binding_site_idxs, pdb_file):

        idxs_pos = pdb_file.xyz[0][binding_site_idxs]
        COM_BS = np.mean(idxs_pos, axis=0)

        return COM_BS

    def sigma_eps_charge_list(self, psf_file, ep_convert):

        sys_sigma = []
        sys_epsilon = []
        sys_charge = []

        for atom in psf_file.atom_list:
            # in units of elementary charge
            sys_charge.append(atom.charge)
            # now in units of nm
            half_rmin = atom.type.rmin*0.1
            sigma = half_rmin*2/2**(1/6)
            sys_sigma.append(sigma)
            # now a positive number in kJ/mol
            sys_epsilon.append(atom.type.epsilon/ep_convert)

        return sys_sigma, sys_epsilon, sys_charge

    def nb_cnb_force_lists(self, system, n_ghosts, n_part_system):
        
        nb_forces = []
        cnb_forces = []

        for i,force in enumerate(system.getForces()):
            force.setForceGroup(i)
            if force.__class__.__name__ == 'NonbondedForce':
                nb_forces.append(force.getForceGroup()) # --> append i, list of indexes
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

        return nb_forces, cnb_forces, exclusion_list

    def generate_init_signals(self, n_ghosts):

        initial_signals = np.zeros((n_ghosts, 4))
        initial_signals[:, 0] = np.random.uniform(low=-0.3, high=+0.3, size=(n_ghosts))
        initial_signals[:, 1] = np.random.uniform(low=0.05, high=0.2, size=(n_ghosts))
        initial_signals[:, 2] = np.random.uniform(low=0.1, high=1.0, size=(n_ghosts))
        initial_signals[:, 3] = 1.0

        return initial_signals

    def add_mlforce(self, system, n_ghosts, n_part_system, target_features):

        # indices of ghost particles in the topology
        ghost_particle_idxs = [gh_idx for gh_idx in range(n_part_system,(n_part_system+n_ghosts))]

        exmlforce = mlforce.PyTorchForce(file=self.ani_model,
                                         targetFeatures=target_features,
                                         particleIndices=ghost_particle_idxs,
                                         signalForceWeights=self.sf_weights,
                                         scale=self.mlforce_scale)

        exmlforce.setForceGroup(self.mlforce_group)
        system.addForce(exmlforce)

        return system, ghost_particle_idxs


    def add_custom_cbf(self, pdb, system, group_num, n_ghosts, ghost_particle_idxs):

        anchor_idxs = []
        for i in range(10,12):
            anchor_idxs.append(pdb.top.residue(81).atom(i).index)

      
        cbf = omm.CustomCentroidBondForce(2, "0.5*k*step(distance(g1,g2) - d0)*(distance(g1,g2) - d0)^2")
        cbf.addGlobalParameter('k', 1000)
        cbf.addGlobalParameter('d0', 0.5)
        anchor_grp_idx = cbf.addGroup(anchor_idxs)
        for gh_idx in range(n_ghosts):
            gh_grp_idx = cbf.addGroup([ghost_particle_idxs[gh_idx]])
            cbf.addBond([anchor_grp_idx, gh_grp_idx])

        cbf.setForceGroup(group_num)
        system.addForce(cbf)
        
        return system

    def add_gs_force(self, system, n_ghosts, group_num, n_part_system,
                     initial_signals, sys_sigma, sys_epsilon, sys_charge,
                     exclusion_list):

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

            # set force parameters
            gs_force.setForceGroup(group_num)
            gs_force.setNonbondedMethod(gs_force.CutoffPeriodic)
            gs_force.setCutoffDistance(1.0)
            system.addForce(gs_force)
        
        return system
    
    def build_system_forces(self):

        target_pos, target_signals, target_features = self.read_target_mol_info(self.target_pkl)
        n_ghosts = target_pos.shape[0]
        initial_signals = self.generate_init_signals(n_ghosts)

        com_bs = self.generate_COM(self.binding_site_idxs, self.pdb)
        pos_arr = np.array(self.crd.positions.value_in_unit(unit.nanometers))
        pdb_pos = np.array([pos_arr])
        init_positions = self.init_positions(com_bs, self.width, n_ghosts, self.min_dist, pdb_pos)
        
        # calculating box length
        box_lengths = pos_arr.max(axis=0) - pos_arr.min(axis=0)
        self.psf.setBox(box_lengths[0] * unit.nanometers,
                   box_lengths[1] * unit.nanometers,
                   box_lengths[2] * unit.nanometers)

        params = read_params(self.toppar_str, self.inputs_path)

        n_part_system = len(self.crd.positions)
        self.crd.positions.extend(unit.quantity.Quantity(init_positions,
                                            unit.nanometers))
        system = self.psf.createSystem(params,
                                  nonbondedMethod=omma.forcefield.CutoffPeriodic,
                                  nonbondedCutoff=1*unit.nanometers,
                                  constraints=omma.forcefield.AllBonds)

        
        psf_ghost_chain = self.psf.topology.addChain(id='G')
        psf_ghost_res = self.psf.topology.addResidue('ghosts',
                                                psf_ghost_chain)
        
        sys_sigma, sys_epsilon, sys_charge = self.sigma_eps_charge_list(self.psf, self.ep_convert)

        # adding ghost particles to the system
        for i in range(n_ghosts):
            system.addParticle(self.ghost_mass)
            self.psf.topology.addAtom('G{0}'.format(i),
                             omma.Element.getBySymbol('Ar'),
                             psf_ghost_res,
                             'G{0}'.format(i))

        nb_forces, cnb_forces, exclusion_list = self.nb_cnb_force_lists(system, n_ghosts, n_part_system)

        # add the mlforce, ccbf, gs_force to the system
        system, ghost_particle_idxs = self.add_mlforce(system, n_ghosts, n_part_system, target_features)
        system = self.add_custom_cbf(self.pdb, system, self.gg_group, n_ghosts, ghost_particle_idxs)
        system = self.add_gs_force(system, n_ghosts, self.sg_group, n_part_system, initial_signals,
                                   sys_sigma, sys_epsilon, sys_charge, exclusion_list)

        return system, initial_signals, n_ghosts, self.psf.topology, self.crd.positions, target_features
