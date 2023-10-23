import sys
import os
import os.path as osp
import pickle as pkl
import numpy as np
import mdtraj as mdj

import openmm.app as omma
import openmm as omm
import openmm.unit as unit

from flexibletopology.utils.openmmutils import read_params, nb_params_from_charmm_psf, add_ghosts_to_system
from flexibletopology.utils.initialize import gen_init_attr, gen_init_pos
from flexibletopology.forces.static import add_ghosts_to_nb_forces
from flexibletopology.forces.dynamic import add_gs_force, add_gg_nb_force

import sys
    
class SystemBuild(object):

    """
    A class that generates a ready-to-equilibrate OpenMM system object.
    """

    def __init__(self, psf=None, pos=None, pdb=None, target_pkl=None, n_ghosts=None, toppar_str=None, inputs_path=None,
                 ani_model=None, width=None, binding_site_idxs=None, min_dist=0.15, gg_min_dist=0.05,
                 sf_weights=None, gg_group=None, gg_nb_group=None, mlforce_group=None, sg_group=None, mlforce_scale=None,
                 ghost_mass=None,attr_bounds=None,assignFreq=None,rmax_delta=None, rest_k=None, contForce=None):

        self.psf = psf
        self.pos = pos
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
        self.sf_weights = sf_weights
        self.gg_group = gg_group
        self.gg_nb_group = gg_nb_group
        self.mlforce_group = mlforce_group
        self.sg_group = sg_group
        self.mlforce_scale = mlforce_scale
        self.ghost_mass = ghost_mass
        self.attr_bounds = attr_bounds
        self.min_dist = min_dist
        self.gg_min_dist = gg_min_dist
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

    def read_target_mol_info(self, mol_file_name):

        with open(mol_file_name, 'rb') as pklf:
            data = pkl.load(pklf)

        return data['target_coords'], data['target_signals'], data['target_features']


    def generate_COM(self, binding_site_idxs, pdb_file):

        idxs_pos = pdb_file.xyz[0][binding_site_idxs]
        COM_BS = np.mean(idxs_pos, axis=0)

        return COM_BS

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


    def add_custom_cbf(self, pdb, system, group_num, ghost_particle_idxs, anchor_idxs):

        cbf = omm.CustomCentroidBondForce(2, "0.5*k*step(distance(g1,g2) - d0)*(distance(g1,g2) - d0)^2")
        cbf.addGlobalParameter('k', 1000) 
        cbf.addGlobalParameter('d0', 0.4) # it was 0.8
        
        anchor_grp_idx = cbf.addGroup(anchor_idxs)
        for gh_idx in range(self.n_ghosts):
            gh_grp_idx = cbf.addGroup([ghost_particle_idxs[gh_idx]])
            cbf.addBond([anchor_grp_idx, gh_grp_idx])

        cbf.setForceGroup(group_num)
        system.addForce(cbf)
        
        return system
    
    def build_system_forces(self):

        if self.target_pkl is not None:
            target_pos, target_signals, target_features = self.read_target_mol_info(self.target_pkl)
            self.n_ghosts = target_pos.shape[0]
        else:
            target_features = None
            
        init_attr = gen_init_attr(self.n_ghosts, self.attr_bounds,total_charge=0,init_lambda=1.0)

        com_bs = self.generate_COM(self.binding_site_idxs, self.pdb)
        init_positions = gen_init_pos(self.n_ghosts, com_bs, self.width, self.pos, self.min_dist, self.gg_min_dist)
        
        # calculating box length
        box_lengths = self.pos.max(axis=0) - self.pos.min(axis=0)
        self.psf.setBox(box_lengths[0] * unit.nanometers,
                   box_lengths[1] * unit.nanometers,
                   box_lengths[2] * unit.nanometers)

        params = read_params(self.toppar_str, self.inputs_path)

        n_part_system = len(self.pos)
        self.pos = np.append(self.pos, init_positions, axis=0)

        system = self.psf.createSystem(params,
                                  nonbondedMethod=omma.forcefield.CutoffPeriodic,
                                  nonbondedCutoff=1*unit.nanometers,
                                  constraints=omma.forcefield.HBonds)

        sys_nb_params = nb_params_from_charmm_psf(self.psf)
        system, new_psf = add_ghosts_to_system(system, self.psf, self.n_ghosts, self.ghost_mass)

        print("Adding ghosts to nb forces")
        system, exclusion_list = add_ghosts_to_nb_forces(system, self.n_ghosts, n_part_system)

        # indices of ghost particles in the topology
        ghost_particle_idxs = list(range(n_part_system,(n_part_system+self.n_ghosts)))
        
        # add the mlforce, ccbf, gs_force to the system
        if self.contforce is not None:
            system.addForce(self.contforce)

        if self.target_pkl is not None:
            system = self.add_mlforce(system, ghost_particle_idxs, target_features)
            
        system = self.add_custom_cbf(self.pdb, system, self.gg_group, ghost_particle_idxs, self.binding_site_idxs)
        system = add_gs_force(system,
                              n_ghosts=self.n_ghosts,
                              n_part_system=n_part_system,
                              initial_attr=init_attr,
                              group_num=self.sg_group,
                              sys_attr=sys_nb_params,
                              nb_exclusion_list=exclusion_list)

        system = add_gg_nb_force(system,
                                 n_ghosts=self.n_ghosts,
                                 n_part_system=n_part_system,
                                 group_num=self.gg_nb_group,
                                 initial_sigmas=init_attr['sigma'],
                                 initial_charges=init_attr['charge'],
                                 nb_exclusion_list=exclusion_list)
    
        return system, init_attr, self.n_ghosts, new_psf.topology, self.pos, target_features


        
