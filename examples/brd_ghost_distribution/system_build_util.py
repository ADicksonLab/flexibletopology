import sys
import os
import os.path as osp
import pickle as pkl
import numpy as np
import mdtraj as mdj

import openmm.app as omma
import openmm as omm
import openmm.unit as unit

from flexibletopology.utils.openmmutils import read_params, nb_params_from_charmm_psf
from flexibletopology.forces.nonbonded import add_gs_force, add_ghosts_to_nb_forces

import sys
    
class SystemBuild(object):

    """
    A class that generates a ready-to-equilibrate OpenMM system object.
    """

    def __init__(self, psf=None, crd=None, pdb=None, target_pkl=None, n_ghosts=None, toppar_str=None, inputs_path=None,
                 ani_model=None, width=None, binding_site_idxs=None, min_dist=0.15, ep_convert=None,
                 sf_weights=None, gg_group=None, mlforce_group=None, sg_group=None, mlforce_scale=None,
                 ghost_mass=None,attr_bounds=None,assignFreq=None,rmax_delta=None, rest_k=None, contForce=None):

        self.psf = psf
        self.crd = crd
        self.pdb = pdb
        self.target_pkl = target_pkl
        self.toppar_str = toppar_str
        self.inputs_path = inputs_path

        if target_pkl is not None:
            assert ani_model is not None, "Must provide a model to compute ani features if using MLForce"
            assert sf_weights is not None, "Must provide sf_weights if using MLForce"
            assert mlforce_scale is not None, "Must provide mlforce_scale if using MLForce"
            assert mlforce_group is not None, "Must provide mlforce_group if using MLForce"
            assert assignFreq is not None, "Must provide assignFreq if using MLForce"
            assert rest_k is not None, "Must provide rest_k if using MLForce"
            assert rmax_delta is not None, "Must provide rmax_delta if using MLForce"
            self.rest_idxs, self.rest_dists = self.getRestraints(target_pkl)
        else:
            assert n_ghosts is not None, "Must provide the number of ghost particles if not using MLForce"
            self.rest_idxs = None
            self.rest_dists = None

        self.n_ghosts = n_ghosts            
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
        self.attr_bounds = attr_bounds
        self.min_dist = min_dist
        self.assignFreq = assignFreq
        self.rmax_delta = rmax_delta
        self.rest_k = rest_k
        self.contforce = contForce
        
        assert binding_site_idxs is not None, "Must give binding site indices"

        
    def getRestraints(self, pkl_name):
        data = pkl.load(open(pkl_name,'rb'))
        bond_mat = data['adjacency']
        coords = data['target_coords']
        n = len(coords)
        bond_idxs = []
        bond_dists = []
        for i in range(n-1):
            for j in range(i+1,n):
                if bond_mat[i,j] == 1:
                    bond_idxs.append([i,j])
                    bond_dists.append(np.sqrt(np.sum(np.square(coords[i]-coords[j]))))

        return bond_idxs, bond_dists

    def getParameters(self, sim, n_ghosts):

        pars = sim.context.getParameters()
        par_array = np.zeros((n_ghosts,5))
        for i in range(n_ghosts):
            tmp_charge = pars[f'charge_g{i}']
            tmp_sigma = pars[f'sigma_g{i}']
            tmp_epsilon = pars[f'epsilon_g{i}']
            tmp_lambda = pars[f'lambda_g{i}']
            tmp_assign = pars[f'assignment_g{i}']
            par_array[i] = np.array([tmp_charge,tmp_sigma,tmp_epsilon,tmp_lambda,tmp_assign])

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

    def generate_init_signals(self, n_ghosts):

        initial_signals = np.zeros((n_ghosts, 4))
        for idx, value in enumerate(self.attr_bounds.values()):
            initial_signals[:, idx] = np.random.uniform(
                low=value[0], high=value[1], size=(n_ghosts))
    
        # set total charge to zero
        total_charge = initial_signals[:, 0].sum()
        initial_signals[:, 0] -= total_charge/n_ghosts
        initial_signals[:, -1] = 0.7 # Do we need this??

        return initial_signals

    def add_mlforce(self, system, ghost_particle_idxs, target_features):

        import mlforce

        init_assign = list(range(len(ghost_particle_idxs)))
        
        exmlforce = mlforce.PyTorchForce(file=self.ani_model,
                                         targetFeatures=mlforce.vectordd(target_features),
                                         particleIndices=mlforce.vectori(ghost_particle_idxs),
                                         signalForceWeights=mlforce.vectord(self.sf_weights),
                                         scale=self.mlforce_scale,
                                         assignFreq=self.assignFreq,
                                         restraintIndices=mlforce.vectorii(self.rest_idxs),
                                         restraintDistances=mlforce.vectord(self.rest_dists),
                                         rmaxDelta=self.rmax_delta,
                                         restraintK=self.rest_k,
                                         initialAssignment=mlforce.vectori(init_assign))

        exmlforce.setForceGroup(self.mlforce_group)
        system.addForce(exmlforce)

        return system


    def add_custom_cbf(self, pdb, system, group_num, n_ghosts, ghost_particle_idxs, anchor_idxs):

        cbf = omm.CustomCentroidBondForce(2, "0.5*k*step(distance(g1,g2) - d0)*(distance(g1,g2) - d0)^2")
        cbf.addGlobalParameter('k', 1000) 
        cbf.addGlobalParameter('d0', 0.4) # it was 0.8
        
        anchor_grp_idx = cbf.addGroup(anchor_idxs)
        for gh_idx in range(n_ghosts):
            gh_grp_idx = cbf.addGroup([ghost_particle_idxs[gh_idx]])
            cbf.addBond([anchor_grp_idx, gh_grp_idx])

        cbf.setForceGroup(group_num)
        system.addForce(cbf)
        
        return system
    
    def build_system_forces(self):

        if self.target_pkl is not None:
            target_pos, target_signals, target_features = self.read_target_mol_info(self.target_pkl)
            n_ghosts = target_pos.shape[0]
        else:
            n_ghosts = self.n_ghosts
            target_features = None
            
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
                                  constraints=omma.forcefield.HBonds)

        
        psf_ghost_chain = self.psf.topology.addChain(id='G')
        psf_ghost_res = self.psf.topology.addResidue('ghosts',
                                                psf_ghost_chain)
        
        sys_nb_params = nb_params_from_charmm_psf(self.psf)

        # adding ghost particles to the system
        for i in range(n_ghosts):
            system.addParticle(self.ghost_mass)
            self.psf.topology.addAtom('G{0}'.format(i),
                             omma.Element.getBySymbol('Ar'),
                             psf_ghost_res,
                             'G{0}'.format(i))

        system, exclusion_list = add_ghosts_to_nb_forces(system, n_ghosts, n_part_system)

        # indices of ghost particles in the topology
        ghost_particle_idxs = list(range(n_part_system,(n_part_system+n_ghosts)))
        
        # add the mlforce, ccbf, gs_force to the system
        if self.contforce is not None:
            system.addForce(self.contforce)

        if self.target_pkl is not None:
            system = self.add_mlforce(system, ghost_particle_idxs, target_features)
            
        system = self.add_custom_cbf(self.pdb, system, self.gg_group, n_ghosts, ghost_particle_idxs, self.binding_site_idxs)
        system = add_gs_force(system, n_ghosts=n_ghosts, n_part_system=n_part_system, initial_signals=initial_signals,
                                   sys_params=sys_nb_params, exclusion_list=exclusion_list)

        return system, initial_signals, n_ghosts, self.psf.topology, self.crd.positions, target_features


        
