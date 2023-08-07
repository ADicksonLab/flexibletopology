import os
import os.path as osp
import numpy as np
import mdtraj as mdj
from wepy.hdf5 import WepyHDF5
from geomm.centering import center_around
from geomm.superimpose import superimpose
from geomm.grouping import group_pair
from geomm.rmsd import calc_rmsd
from geomm.box_vectors import box_vectors_to_lengths_angles
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import sys
import statistics
warnings.filterwarnings("ignore")

def save_fig( fig_id , IMAGES_PATH, fig_extension="png"):
    os.makedirs(IMAGES_PATH, exist_ok=True)
    plt.rcParams["figure.dpi"] = 200
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    plt.savefig(path, format=fig_extension)
    

def compute_assembly_rmsd(fields, ref_pdb, ref_bb_idxs, ref_lig_idxs, align_bb_idxs, align_lig_idxs, align_all_idxs, cycles , **kwargs):
    """ Returns assembly RMSD.

    Parameters
    ----------

    ref_pdb: str
        reference pdb file name

    ref_bb_idxs : arraylike of int
        The reference backbone idxs.

    ref_lig_idxs : arraylike of int
        The reference ligand idxs.

    align_bb_idxs : arraylike of int
        The backbone idxs of the system that will be rotated to match the ref_coords.

    align_lig_idxs : arraylike of int
        The ligand idxs of the system that will be rotated to match the ref_coords.

    cycles : int
        Number of cycles in each walker.
    
    """

    pos = fields['positions']
    bvs = fields['box_vectors']
    assignments = fields['assignments'].astype(int)
    box_lengths, box_angles = box_vectors_to_lengths_angles(box_vectors = bvs)
    lig_idxs = np.array(range(len(ref_bb_idxs),len(ref_bb_idxs)+len(ref_lig_idxs)))
    assembly_rmsds = []

    for cycle in range(cycles):

        grouped_walker_pos = group_pair(coords = pos[cycle,align_all_idxs,:] ,
                                        unitcell_side_lengths = box_lengths[cycle] ,
                                        member_a_idxs = align_bb_idxs,
                                        member_b_idxs = align_lig_idxs)

        # which ghost atoms correspond to non-hydrogen atoms
        non_h_ghosts = [i + len(align_bb_idxs) for i,a in enumerate(assignments[cycle]) if a < len(ref_lig_idxs)]
        non_h_assignments = [a for a in assignments[cycle] if a < len(ref_lig_idxs)]

        to_keep = align_bb_idxs.tolist() + non_h_ghosts

        centered_walker_pos = center_around(coords = grouped_walker_pos[to_keep],
                                            idxs = lig_idxs)
        
        # this has only non-h ghost atoms
        mapped_ref_lig = ref_lig_idxs[non_h_assignments]
        mapped_ref_all_idxs = ref_bb_idxs.tolist() + mapped_ref_lig.tolist()
        mapped_ref_pos = ref_pdb.xyz[0,mapped_ref_all_idxs,:]

        centered_mapped_ref_pos = center_around(coords = mapped_ref_pos,
                                                idxs = lig_idxs)
        
        sup_walker_pos,rotation_matrix,qcp_rmsd = superimpose(ref_coords = centered_mapped_ref_pos,
                                                            coords = centered_walker_pos,
                                                            idxs = lig_idxs)
                    
        assembly_rmsd = calc_rmsd(ref_coords = centered_mapped_ref_pos,
                                coords = sup_walker_pos,
                                idxs = lig_idxs)

        assembly_rmsds.append(assembly_rmsd)

    return np.array(assembly_rmsds)

def compute_pose_rmsd(fields, ref_pdb, ref_bb_idxs, ref_lig_idxs, align_bb_idxs, align_lig_idxs, align_all_idxs, cycles , **kwargs):

    """ Returns pose RMSD.

    Parameters
    ----------

    ref_pdb: str
        reference pdb file name

    ref_bb_idxs : arraylike of int
        The reference backbone idxs.

    ref_lig_idxs : arraylike of int
        The reference ligand idxs.

    align_bb_idxs : arraylike of int
        The backbone idxs of the system that will be rotated to match the ref_coords.

    align_lig_idxs : arraylike of int
        The ligand idxs of the system that will be rotated to match the ref_coords.

    cycles : int
        Number of cycles in each walker.
    
    """
    
    pos = fields['positions']
    bvs = fields['box_vectors']
    assignments = fields['assignments'].astype(int)
    box_lengths, box_angles = box_vectors_to_lengths_angles(box_vectors = bvs)
    lig_idxs = np.array(range(len(ref_bb_idxs),len(ref_bb_idxs)+len(ref_lig_idxs)))
    pose_rmsds = []

    for cycle in range(cycles):

        grouped_walker_pos = group_pair(coords = pos[cycle,align_all_idxs,:] ,
                                        unitcell_side_lengths = box_lengths[cycle] ,
                                        member_a_idxs = align_bb_idxs,
                                        member_b_idxs = align_lig_idxs)

        # which ghost atoms correspond to non-hydrogen atoms
        non_h_ghosts = [i + len(align_bb_idxs) for i,a in enumerate(assignments[cycle]) if a < len(ref_lig_idxs)]
        non_h_assignments = [a for a in assignments[cycle] if a < len(ref_lig_idxs)]

        to_keep = align_bb_idxs.tolist() + non_h_ghosts

        centered_walker_pos = center_around(coords = grouped_walker_pos[to_keep],
                                            idxs = align_bb_idxs)

        mapped_ref_lig = ref_lig_idxs[non_h_assignments]
        mapped_ref_all_idxs = ref_bb_idxs.tolist() + mapped_ref_lig.tolist()
        mapped_ref_pos = ref_pdb.xyz[0,mapped_ref_all_idxs,:]

        centered_mapped_ref_pos = center_around(coords = mapped_ref_pos,
                                                idxs = range(len(ref_bb_idxs)))
        
        sup_walker_pos,rotation_matrix,qcp_rmsd= superimpose(ref_coords = centered_mapped_ref_pos,
                                                             coords = centered_walker_pos,
                                                             idxs = range(len(ref_bb_idxs)))

        pose_rmsd = calc_rmsd(ref_coords = centered_mapped_ref_pos,
                            coords = sup_walker_pos,
                            idxs = lig_idxs)
            
        pose_rmsds.append(pose_rmsd)


    return np.array(pose_rmsds)


def compute_aligned_pos(fields, ref_pdb, ref_bb_idxs, ref_lig_idxs, align_bb_idxs, align_lig_idxs, align_all_idxs, cycles, **kwargs):

    """Returns ligned positions of ghost particles that are re-ordered using the
    assignments.
    
    Parameters
    ----------

    ref_pdb: str
        reference pdb file name

    ref_bb_idxs : arraylike of int
        The reference backbone idxs.

    ref_lig_idxs : arraylike of int
        The reference ligand idxs.

    align_bb_idxs : arraylike of int
        The backbone idxs of the system that will be rotated to match the ref_coords.

    align_lig_idxs : arraylike of int
        The ligand idxs of the system that will be rotated to match the ref_coords.

    cycles : int
        Number of cycles in each walker.
        
    """

    pos = fields['positions']
    bvs = fields['box_vectors']
    assignments = fields['assignments'].astype(int)
    box_lengths, box_angles = box_vectors_to_lengths_angles(box_vectors = bvs)
    all_pos = []
    
    for cycle in range(cycles):

        grouped_walker_pos = group_pair(coords = pos[cycle,align_all_idxs,:] ,
                                        unitcell_side_lengths = box_lengths[cycle] ,
                                        member_a_idxs = align_bb_idxs,
                                        member_b_idxs = align_lig_idxs)

        # which ghost atoms correspond to non-hydrogen atoms
        non_h_ghosts = [i + len(align_bb_idxs) for i,a in enumerate(assignments[cycle]) if a < len(ref_lig_idxs)]
        non_h_assignments = [a for a in assignments[cycle] if a < len(ref_lig_idxs)]

        to_keep = align_bb_idxs.tolist() + non_h_ghosts

        centered_walker_pos = center_around(coords = grouped_walker_pos[to_keep],
                                            idxs = align_bb_idxs)

        mapped_ref_lig = ref_lig_idxs[non_h_assignments]
        mapped_ref_all_idxs = ref_bb_idxs.tolist() + mapped_ref_lig.tolist()
        mapped_ref_pos = ref_pdb.xyz[0,mapped_ref_all_idxs,:]

        centered_mapped_ref_pos = center_around(coords = mapped_ref_pos,
                                                idxs = range(len(ref_bb_idxs)))
        
        sup_walker_pos,rotation_matrix,qcp_rmsd = superimpose(ref_coords = centered_mapped_ref_pos,
                                                             coords = centered_walker_pos,
                                                             idxs = range(len(ref_bb_idxs)))

        # reorder the sup_walker_pos using the assignments
        reorder_walker_pos = np.zeros((len(ref_lig_idxs),3))

        for i in range (len(ref_lig_idxs)):
            # reorder_walker_pos[i] is the ghost atom corresponding to ref atom i
            gh_idx = non_h_assignments.index(i)
            reorder_walker_pos[i] = sup_walker_pos[len(align_bb_idxs) + gh_idx]


        all_pos.append(reorder_walker_pos)

    return np.array(all_pos)


class FTAnalysisEngine(object):

    def __init__(self, base_dir, filename, mode , ref_pdb, ref_bb_idxs, ref_lig_idxs, align_bb_idxs, align_lig_idxs , align_all_idxs, **kwargs):
        self.base_dir = base_dir
        self._h5 = WepyHDF5(filename=filename,mode=mode)  
        self.ref_pdb = ref_pdb
        self.ref_bb_idxs = ref_bb_idxs
        self.ref_lig_idxs = ref_lig_idxs
        self.align_bb_idxs = align_bb_idxs
        self.align_lig_idxs = align_lig_idxs
        self.align_all_idxs = align_all_idxs
        self.has_been_called = {}
        with self._h5:
            self.runs = self._h5.num_runs
            self.cycles = self._h5.num_run_cycles(0)
            self.walkers = self._h5.num_trajs 
            self.num_atoms = self._h5.num_atoms
        
    """Calculate the assembly and pose RMSD between the reference and query coordinates

    Parameters
    ----------

    base_dir : str
        Base directory to save the output files and graphs

    filename : str
        File path

    mode : str
        Mode specification for opening the HDF5 file.

    ref_bb_idxs : arraylike of int
        The reference backbone idxs.

    ref_lig_idxs : arraylike of int
        The reference ligand idxs.

    align_bb_idxs : arraylike of int
        The backbone idxs of the system that will be rotated to match the ref_coords.

    align_lig_idxs : arraylike of int
        The ligand idxs of the system that will be rotated to match the ref_coords.

    """

    def assembly_rmsd( self ,  pdbfile , field_tag):
        """Compute assembly rmsds and save to the self._h5 file.

        Parameters
        ----------

        pdbfile : str
            pdbfile name

        field_tag : None or string
            If not None, a string that specifies the name of the
            observables sub-field that the computed values will be saved to.

        """

        self.has_been_called["assembly_rmsd"] = True
        
        with self._h5:
            assembly_rmsd_list = self._h5.compute_observable(compute_assembly_rmsd,
                                                        ['positions','box_vectors','assignments'] ,
                                                        args = (( pdbfile , self.ref_bb_idxs , self.ref_lig_idxs , self.align_bb_idxs , self.align_lig_idxs ,  self.align_all_idxs , self.cycles )),
                                                        save_to_hdf5 = field_tag
                                                        )

        self.assembly_rmsd_array = np.array(assembly_rmsd_list)
        return self.assembly_rmsd_array

    def pose_rmsd(self , pdbfile , field_tag ):
        """Compute pose rmsds and save to the self._h5 file.

        Parameters
        ----------

        pdbfile : str
            pdbfile name

        field_tag : None or string
            If not None, a string that specifies the name of the
            observables sub-field that the computed values will be saved to.

        """
        self.has_been_called["pose_rmsd"] = True
         
        with self._h5:
            pose_rmsd_list = self._h5.compute_observable(compute_pose_rmsd,
                                                        ['positions','box_vectors','assignments'] ,
                                                        args = (( pdbfile , self.ref_bb_idxs , self.ref_lig_idxs , self.align_bb_idxs , self.align_lig_idxs ,  self.align_all_idxs , self.cycles )),
                                                        save_to_hdf5 = field_tag
                                                        )
            
        self.pose_rmsd_array = np.array(pose_rmsd_list)
        return self.pose_rmsd_array
    

    def aligned_pos(self , pdbfile , field_tag ): 
        """Compute ligand pos and save to the self._h5 file.

        Parameters
        ----------

        pdbfile : str
            pdbfile name

        field_tag : None or string
            If not None, a string that specifies the name of the
            observables sub-field that the computed values will be saved to.

        """
        self.has_been_called["aligned_pos"] = True
        
        with self._h5:
            aligned_pos_list = self._h5.compute_observable(compute_aligned_pos,
                                                            ['positions','box_vectors','assignments'],
                                                            args = ((pdbfile , self.ref_bb_idxs , self.ref_lig_idxs , self.align_bb_idxs , self.align_lig_idxs ,  self.align_all_idxs , self.cycles)),
                                                            save_to_hdf5 = field_tag
                                                            )

            
        self.aligned_pos_array = np.array(aligned_pos_list)
        return self.aligned_pos_array
    
    def calculate_loss_per_particle (self , mlforce_scale):
        """Calculate loss per number of particles.

        Parameters
        ----------

        mlforce_scale : string
            path to the mlforce_scale force list (.txt) file.
            
        """

        self.has_been_called["calculate_loss"] = True

        with self._h5:

            assignments = np.array(self._h5.get_traj_field(0,0,'assignments'))
            force_potential_energies = []
            for walker in range(self.walkers):
                force_potential_energy = np.array(self._h5.get_traj_field(0,walker,'force_potential_energy'))
                force_potential_energies.append(force_potential_energy)

        force_potential_energies = np.array(force_potential_energies).reshape(self.walkers, self.cycles)
        mlforce_scale = np.loadtxt(mlforce_scale)
        num_particles = assignments.shape[1]
        self.loss = force_potential_energies / (mlforce_scale * num_particles)

        return self.loss
    
    
    def rmsd_analysis(self, assembly_rmsd_cutoff , pose_rmsd_cutoff, print_results = True):

        """analyze the calculated assembly and pose RMSDs.

        Parameters
        ----------

        assembly_rmsd_cutoff : float 
            value of assembly_rmsd_cutoff in nm.

        pose_rmsd_cutoff : float 
            value of pose_rmsd_cutoff in nm.
        
        print-results : bool , defult = True
        
        """

        if any(k not in self.has_been_called for k in ('assembly_rmsd' , 'pose_rmsd' )) :

            print("Please call assembly_rmsd(), pose_rmsd() methods before calling rmsd_analysis()")
            return None
            
        else:

            # calculate the minimum pose rmsd over the whole trajectories and its it's walker and cycle index
            min_pose_rmsd = np.min(self.pose_rmsd_array)
            w_min_pose, c_min_pose = np.where(self.pose_rmsd_array == min_pose_rmsd)
            min_pose_rmsd_walker_index = w_min_pose[0]
            min_pose_rmsd_cycle_index = c_min_pose[0]
            assembly_rmsd_of_min_pose_rmsd = self.assembly_rmsd_array[min_pose_rmsd_walker_index , min_pose_rmsd_cycle_index]

            min_assembly_rmsd = np.min(self.assembly_rmsd_array)
            w_min_assembly, c_min_assembly = np.where(self.assembly_rmsd_array == min_assembly_rmsd)
            min_assembly_rmsd_walker_index = w_min_assembly[0]
            min_assembly_rmsd_cycle_index =  c_min_assembly[0]
            pose_rmsd_of_min_assembly_rmsd = self.pose_rmsd_array[min_assembly_rmsd_walker_index , min_assembly_rmsd_cycle_index]



            # obtain valid frames lig pos and idxs based on the assembly RMSD
            valid_frames = self.assembly_rmsd_array < assembly_rmsd_cutoff #valid_poses
           

            # calculate the percentage of frames with valid assembly and pose RMSD in each walker
            num_frames_valid_assembly, num_frames_valid_pose = [] , []

            for i in range(self.walkers):

                valid_assembly_idxs = np.where(self.assembly_rmsd_array[i] < assembly_rmsd_cutoff)
                valid_pose_idxs = np.where(self.pose_rmsd_array[i] < pose_rmsd_cutoff)
                num_frames_valid_assembly.append(np.shape(valid_assembly_idxs)[1])
                num_frames_valid_pose.append(np.shape(valid_pose_idxs)[1])
            
            percentage_valid_assembly = 100 * np.array(num_frames_valid_assembly)/self.cycles
            percentage_valid_pose = 100 * np.array(num_frames_valid_pose)/self.cycles

            # calculate the percentage of frames with valid assembly or pose RMSD over the whole trajectories
            num_total_valid_assembly = np.shape(np.where(valid_frames))[1]
            num_total_valid_pose = np.shape(np.where(self.pose_rmsd_array < pose_rmsd_cutoff))[1]
            percentage_total_valid_pose = 100 * np.array(num_total_valid_pose)/(self.cycles * self.walkers)
            percentage_total_valid_assembly = 100 * np.array(num_total_valid_assembly)/(self.cycles * self.walkers)

            # calculate the percentage of the simulations that have at least 1 frames with valid assembly or pose RMSD
            percentage_valid_assembly_sim = 100*(np.count_nonzero(np.array(num_frames_valid_assembly) > 0))/self.walkers 
            percentage_valid_pose_sim = 100*(np.count_nonzero(np.array(num_frames_valid_pose) > 0))/(self.walkers)

            rmsd_analysis_summary = {'title' : "rmsd_analysis_summary",
                                    'percentage_valid_assembly' : percentage_valid_assembly,
                                    'percentage_valid_pose' : percentage_valid_pose,
                                    'percentage_total_valid_assembly' : percentage_total_valid_assembly,
                                    'percentage_total_valid_pose' : percentage_total_valid_pose,
                                    'percentage_valid_assembly_simulations' : percentage_valid_assembly_sim,
                                    'percentage_valid_pose_simulations' :  percentage_valid_pose_sim,
                                    'min_pos_rmsd' : min_pose_rmsd,
                                    'min_pose_rmsd_walker_index' : min_pose_rmsd_walker_index,
                                    'min_pose_rmsd_cycle_index' : min_pose_rmsd_cycle_index,
                                    'assembly_rmsd_of_min_pose_rmsd' : assembly_rmsd_of_min_pose_rmsd,
                                    'min_assembly_rmsd' : min_assembly_rmsd,
                                    'min_assembly_rmsd_walker_index' : min_assembly_rmsd_walker_index,
                                    'min_assembly_rmsd_cycle_index' : min_assembly_rmsd_cycle_index,
                                    'pose_rmsd_of_min_assembly_rmsd' : pose_rmsd_of_min_assembly_rmsd
                                    }

            if (print_results):
                for key, value in rmsd_analysis_summary.items():
                    print(key, ':' , value)
       

            return rmsd_analysis_summary
        
    
    def optimal_num_clusters(self, assembly_rmsd_cutoff , num_clusters, draw_plot = True ):
        """Find the optimal number of clusters by calculating the silhouette scores for different number of clusters.
        The optimal number of clusters k is the one that maximize the average silhouette over a range of possible values for k.

        Parameters
        ----------

        assembly_rmsd_cutoff : float 
            Assembly rmsd cutoff value in nm.

        num_clusters : arraylike of int
            Number of clusters to find the optimal number of clusters from.

        draw-plot : bool, default = True
            If true, saves the inertia and sillhoutte scores Vs. k.

        """ 
        
        if any(k not in self.has_been_called for k in ('assembly_rmsd' , 'aligned_pos')) :

            print("Please call assembly_rmsd() and aligned_pos() methods before calling optimal_num_clusters()")
            return None
            

        else:

            valid_frames = self.assembly_rmsd_array < assembly_rmsd_cutoff 
            valid_aligned_pos = self.aligned_pos_array[valid_frames]

            num_frames = np.shape(valid_aligned_pos)[0]
            self.num_atoms = np.shape(valid_aligned_pos)[1]
            valid_aligned_pos_reshaped = valid_aligned_pos.reshape(num_frames, self.num_atoms * 3)
            
            
            # find the optimal number of clusters
            kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(valid_aligned_pos_reshaped)
                            for k in num_clusters]

            # Silhouette score
            silhouette_scores = [silhouette_score(valid_aligned_pos_reshaped, model.labels_)
                        for model in kmeans_per_k]
            
            self.ss_optimal_num_clusters = num_clusters[np.argmax(silhouette_scores)]
            optimal_num_clusters_summary = {'ss_optimal_num_clusters' : self.ss_optimal_num_clusters,
                                    'ss_scores' : silhouette_scores,
                                    }
            # 
            if (draw_plot):
                plot_path = os.path.join(self.base_dir, "plots", "clustering")
                os.makedirs(plot_path, exist_ok=True)

                # 1st plot : elbow 
                inertias = [model.inertia_ for model in kmeans_per_k]
                plt.figure(figsize=(8, 3.5))
                plt.plot(num_clusters, inertias, "bo-")
                plt.xlabel("$k$", fontsize=14)
                plt.ylabel("Inertia", fontsize=14)
                plt.xticks(num_clusters)
                
                save_fig("inertia_vs_k_plot" , plot_path )
            
                

                # 2nd plot : Silhouette score
                plt.figure(figsize=(8, 3))
                plt.plot(num_clusters, silhouette_scores, "bo-")
                plt.xlabel("$k$", fontsize=12)
                plt.ylabel("Silhouette score", fontsize=12)
                plt.xticks(num_clusters)
                save_fig( "silhouette_score_vs_k_plot" , plot_path )
                return optimal_num_clusters_summary
           
    
    
        
    def clustering( self, assembly_rmsd_cutoff ): #,  num_clusters = None ):

        """clustering the ghost particles positions.                                                                  
        Parameters
        ----------

        assembly_rmsd_cutoff : float 
            Assembly rmsd cutoff value in nm.

        num_clusters : int , optional
            Number of clusters. Default = ss_optimal_num_clusters
        
        """ 
        self.has_been_called["clustering"] = True

        if any(k not in self.has_been_called for k in ('assembly_rmsd' , 'aligned_pos')) :

            print("Please call assembly_rmsd() and aligned_pos() methods before calling clustering()")
            return None
            
        else:

            # if num_clusters is None:
            #     num_clusters = self.optimal_num_clusters
                

            valid_frames = self.assembly_rmsd_array < assembly_rmsd_cutoff 

            self.valid_aligned_pos = self.aligned_pos_array[valid_frames]
            self.valid_walker_idxs , self.valid_cycle_idxs = np.where(valid_frames)

            num_frames = np.shape(self.valid_aligned_pos)[0]
            num_clusters = int(num_frames / 10)
            self.num_atoms = np.shape(self.valid_aligned_pos)[1]
            valid_aligned_pos_reshaped = self.valid_aligned_pos.reshape(num_frames, self.num_atoms * 3)
            
            # find the optimal number of clusters
            kmeans = KMeans(n_clusters=num_clusters, n_init = 10, random_state=42).fit(valid_aligned_pos_reshaped)
            cluster_centers = (kmeans.cluster_centers_).reshape(num_clusters,self.num_atoms,3)
            labels = kmeans.labels_

            # obtain the most populated cluster center                                                                               
            population = []
            for i in np.unique(labels):
                pop = len(np.where(labels == i)[0])
                population.append(pop)


            most_pop_cluster_index = np.argmax(population)
            self.most_pop_cluster_center = cluster_centers[most_pop_cluster_index].reshape(self.num_atoms,3)
            self.frame_idxs = np.where(labels == most_pop_cluster_index)[0]

            return self.most_pop_cluster_center , self.frame_idxs

    def find_centroid(self, print_results = True ): 
        """Calculate the centroid ghost particles positions among the valid farmes.
        Return the centroid summary, and save the centroid pdb.

        parameters
        ----------
        print_results : bool, defult = True
 
        """
        if any(k not in self.has_been_called for k in ('assembly_rmsd' , 'aligned_pos' , 'pose_rmsd' , 'clustering')) :

            print("Please call assembly_rmsd(), pose_rmsd(), aligned_pos(), clustering() methods before calling find_centroid()")
            return None
            
        else:
        
            distances = ave_distances  = np.zeros(len(self.frame_idxs))
            
            for i,v in enumerate(self.frame_idxs):
                for j in range(self.num_atoms):
                    distances [i] += distance.euclidean(self.most_pop_cluster_center[j], self.valid_aligned_pos[v][j])
                ave_distances [i] = distances [i] / self.num_atoms

            
            min_ave_distance = np.amin(ave_distances)
            centroid_frame_index = self.frame_idxs[np.where(ave_distances == min_ave_distance)]
            centroid_pos = self.valid_aligned_pos[centroid_frame_index].reshape(self.num_atoms,3)
            centroid_walker_index = self.valid_walker_idxs[centroid_frame_index][0]
            centroid_cycle_index = self.valid_cycle_idxs[centroid_frame_index][0]
            centroid_assembly_rmsd =  np.array(self.assembly_rmsd_array)[centroid_walker_index , centroid_cycle_index]
            centroid_pose_rmsd = np.array(self.pose_rmsd_array)[centroid_walker_index , centroid_cycle_index]

            centroid_summary = {'title': "centroid_summary",
                                'min_ave_distance' : min_ave_distance,
                                'walker' : centroid_walker_index,
                                'cycle' : centroid_cycle_index,
                                'assembly_rmsd': centroid_assembly_rmsd,
                                'pose_rmsd' : centroid_pose_rmsd}

            # save the pdb of centroid                                                                                              
            centered_lig_idxs = self.ref_pdb.top.select("chainid 1")
            centroid_pdb = mdj.Trajectory(centroid_pos , self.ref_pdb.atom_slice(centered_lig_idxs).top)
            centroid_pdb.save_pdb(osp.join(self.base_dir,f'centroid_pdb.pdb'))

            if (print_results):
                print("Centroid Summary:")
                for key, value in centroid_summary.items():
                    print(key, ':' , value)
            # Open a file for writing
            summary_path = osp.join(self.base_dir, "summary.txt")
            with open(summary_path, 'w') as file:
                for key, value in centroid_summary.items():
                    line = f"{key}: {value}\n"
                    file.write(line)
           

            return centroid_summary
    
    def plot (self):
        """Plot assembly and pose RMSD, and loss per particles graphs."""


        if any(k not in self.has_been_called for k in ('assembly_rmsd' , 'pose_rmsd' , 'calculate_loss')) :

            print("Please call assembly_rmsd(), pose_rmsd(), and calculate_loss_per_particle methods before calling plot()")
            return None    

        else:

            plot_path = osp.join(self.base_dir,'plots')
            os.makedirs(plot_path, exist_ok=True)

            cycles_array= range(0,self.cycles,1)
            ave_pose_rmsd = np.mean(self.pose_rmsd_array,axis=0)
            ave_assembly_rmsd = np.mean(self.assembly_rmsd_array,axis=0)

            assemblyRMSD_fig = plt.figure(figsize = (5,5))
            for walker in range (self.walkers):
                plt.plot(cycles_array,self.assembly_rmsd_array[walker],color='gray',ls='-',lw=3 ,  alpha = 0.3)
            plt.plot(cycles_array,ave_assembly_rmsd,color='black',ls='-',lw=3 , label = 'Assembly RMSD')
            
            plt.xlabel("Cycle",fontsize=12,fontweight='bold',color='k')
            plt.ylabel("Assembly RMSD (nm)",fontsize=12,fontweight='bold',color='k')
            plt.xticks(np.arange(0,100.1 , step = 20))
            plt.yticks(np.arange(0,0.51 , step = 0.1))
            plt.legend()
            save_fig(f'assembly_rmsd' , plot_path )
            plt.show

            poseRMSD_fig = plt.figure(figsize = (5,5))
            for walker in range (self.walkers):
                plt.plot(cycles_array,self.pose_rmsd_array[walker],color='gray',ls='-',lw=3 ,  alpha = 0.3)
            plt.plot(cycles_array,ave_pose_rmsd,color='black',ls='-',lw=3 , label = 'Pose RMSD')

            plt.xlabel("Cycle",fontsize=12,fontweight='bold',color='k')
            plt.ylabel("Pose RMSD (nm)",fontsize=12,fontweight='bold',color='k')
            plt.xticks(np.arange(0,100.1 , step = 20))
            plt.yticks(np.arange(0,0.71 , step = 0.1))
            plt.legend()
            save_fig(f'pose_rmsd' , plot_path )
            plt.show

            ave_loss = np.mean(self.loss,axis=0)
        

            loss_fig = plt.figure(figsize = (5,5))
            for walker in range (self.walkers):
                plt.plot(cycles_array,self.loss[walker],color='gray',ls='-',lw=3 ,  alpha = 0.3)
            plt.plot(cycles_array,ave_loss,color='black',ls='-',lw=3 , label = 'Loss per Particle')

            plt.xlabel("Cycle",fontsize=12,fontweight='bold',color='k')
            plt.ylabel("Loss per Particle",fontsize=12,fontweight='bold',color='k')
            plt.legend()
            save_fig(f'loss' , plot_path )
            plt.show

            loss_RMSD_fig = plt.figure(figsize = (5,5))
            for walker in range (self.walkers):
                plt.plot(self.assembly_rmsd_array[walker],self.loss[walker], 'o' , color='gray', alpha = 0.3)
            
            plt.xlabel("RMSD",fontsize=12,fontweight='bold',color='k')
            plt.ylabel("Loss per Particle",fontsize=12,fontweight='bold',color='k')
            plt.yscale('log')
            save_fig(f'loss_RMSD' , plot_path )
            plt.show
        
    def rmsd_analysis_multiple_hdf5(self, run_nums, assembly_rmsds_array, pose_rmsds_array, assembly_rmsd_cutoff , pose_rmsd_cutoff, print_results = True):

        """analyze the calculated assembly and pose RMSDs of multiple runs with the same number of walkers.

        Parameters
        ----------
        run_nums : list of int
            list of the run numbers.
        
        assembly_rmsds_array : ArrayLike of float
            Assembly rmsds of all simulations (e.g. shape of 1: (8,100) shape of 2 simulations should be: (16,100))

        pose_rmsds_array : ArrayLike of float
            pose rmsds of all simulations (e.g. shape of 1: (8,100) shape of 2 simlations should be: (16,100))

        assembly_rmsd_cutoff : float 
            value of assembly_rmsd_cutoff in nm.

        pose_rmsd_cutoff : float 
            value of pose_rmsd_cutoff in nm.
        
        print-results : bool , defult = True
        
        """

        if any(k not in self.has_been_called for k in ('assembly_rmsd' , 'pose_rmsd' )) :

            print("Please call assembly_rmsd(), pose_rmsd() methods before calling rmsd_analysis()")
            return None
            
        else:
            self.has_been_called["rmsd_analysis_multi_hdf5"] = True
            
        
            self.run_nums = run_nums
            self.assembly_rmsds_array = assembly_rmsds_array
            self.pose_rmsds_array = pose_rmsds_array
            self.walkers_multi_hdf5 = assembly_rmsds_array.shape[0]

            # calculate the minimum pose rmsd over the whole trajectories and its it's walker and cycle index
            min_pose_rmsd = np.min(self.pose_rmsds_array)
            w_min_pose, c_min_pose = np.where(self.pose_rmsds_array == min_pose_rmsd)
            min_pose_rmsd_c_index = c_min_pose[0]
            assembly_rmsd_of_min_pose_rmsd = self.assembly_rmsds_array[w_min_pose[0] , min_pose_rmsd_c_index]

            min_pose_rmsd_run_index, min_pose_rmsd_final_w_index = divmod(w_min_pose[0], self.walkers)
            min_pose_rmsd_final_run_index = self.run_nums[min_pose_rmsd_run_index]  

            min_assembly_rmsd = np.min(self.assembly_rmsds_array)
            w_min_assembly, c_min_assembly = np.where(self.assembly_rmsds_array == min_assembly_rmsd)
            min_assembly_rmsd_c_index =  c_min_assembly[0]
            pose_rmsd_of_min_assembly_rmsd = self.pose_rmsds_array[w_min_assembly[0] , min_assembly_rmsd_c_index]
            min_assembly_rmsd_run_index, min_assembly_rmsd_final_w_index = divmod(w_min_assembly[0], self.walkers)
            min_assembly_rmsd_final_run_index = self.run_nums[min_assembly_rmsd_run_index]  

            # obtain valid frames lig pos and idxs based on the assembly RMSD
            valid_frames = self.assembly_rmsds_array < assembly_rmsd_cutoff #valid_poses
           
            # calculate the percentage of frames with valid assembly and pose RMSD in each walker
            num_frames_valid_assembly, num_frames_valid_pose = [] , []

            for i in range(assembly_rmsds_array.shape[0]):

                valid_assembly_idxs = np.where(self.assembly_rmsds_array[i] < assembly_rmsd_cutoff)
                valid_pose_idxs = np.where(self.pose_rmsds_array[i] < pose_rmsd_cutoff)
                num_frames_valid_assembly.append(np.shape(valid_assembly_idxs)[1])
                num_frames_valid_pose.append(np.shape(valid_pose_idxs)[1])
            
            percentage_valid_assembly = 100 * np.array(num_frames_valid_assembly)/self.cycles
            percentage_valid_pose = 100 * np.array(num_frames_valid_pose)/self.cycles

            # calculate the percentage of frames with valid assembly or pose RMSD over the whole trajectories
            num_total_valid_assembly = np.shape(np.where(valid_frames))[1]
            num_total_valid_pose = np.shape(np.where(self.pose_rmsd_array < pose_rmsd_cutoff))[1]
            percentage_total_valid_pose = 100 * np.array(num_total_valid_pose)/(self.cycles * self.walkers_multi_hdf5)
            percentage_total_valid_assembly = 100 * np.array(num_total_valid_assembly)/(self.cycles * self.walkers_multi_hdf5)

            # calculate the percentage of the simulations that have at least 1 frames with valid assembly or pose RMSD
            percentage_valid_assembly_sim = 100*(np.count_nonzero(np.array(num_frames_valid_assembly) > 0))/self.walkers_multi_hdf5
            percentage_valid_pose_sim = 100*(np.count_nonzero(np.array(num_frames_valid_pose) > 0))/self.walkers_multi_hdf5

            self.rmsd_analysis_summary_multi_hdf5 = {'title' : "rmsd_analysis_multi_hdf5_summary",
                                    'percentage_valid_assembly' : percentage_valid_assembly,
                                    'percentage_valid_pose' : percentage_valid_pose,
                                    'percentage_total_valid_assembly' : percentage_total_valid_assembly,
                                    'percentage_total_valid_pose' : percentage_total_valid_pose,
                                    'percentage_valid_assembly_simulations' : percentage_valid_assembly_sim,
                                    'percentage_valid_pose_simulations' :  percentage_valid_pose_sim,
                                    'min_pos_rmsd' : min_pose_rmsd,
                                    'min_pose_rmsd_run_index' : min_pose_rmsd_final_run_index,
                                    'min_pose_rmsd_walker_index' : min_pose_rmsd_final_w_index,
                                    'min_pose_rmsd_cycle_index' : min_pose_rmsd_c_index,
                                    'assembly_rmsd_of_min_pose_rmsd' : assembly_rmsd_of_min_pose_rmsd,
                                    'min_assembly_rmsd' : min_assembly_rmsd,
                                    'min_assembly_rmsd_run_index' : min_assembly_rmsd_final_run_index,
                                    'min_assembly_rmsd_walker_index' : min_assembly_rmsd_final_w_index,
                                    'min_assembly_rmsd_cycle_index' : min_assembly_rmsd_c_index,
                                    'pose_rmsd_of_min_assembly_rmsd' : pose_rmsd_of_min_assembly_rmsd
                                    }     

        if (print_results):
            for key, value in self.rmsd_analysis_summary_multi_hdf5.items():
                print(key, ':' , value)
            
            

        return self.rmsd_analysis_summary_multi_hdf5
        
    def clustering_multi_hdf5( self, assembly_rmsds_array , pose_rmsds_array, aligned_poses_array , assembly_rmsd_cutoff ): #,  num_clusters = None ):

        """clustering the ghost particles positions of multiple runs with the same number of walkers.                                                                  
        Parameters
        ----------

        assembly_rmsds_array : ArrayLike of float
            Assembly rmsds of all simulations (e.g. shape of 1: (8,100) shape of 2 simulations should be: (16,100))

        pose_rmsds_array : ArrayLike of float
            pose rmsds of all simulations (e.g. shape of 1: (8,100) shape of 2 simlations should be: (16,100))
        
        aligned_poses_array: ArrayLike of float
            aligend poses of ghost particles of all simulations (e.g. shape of 1: (8,100,7,3) shape of 2 simulations should be: (16,100,7,3))


        assembly_rmsd_cutoff : float 
            Assembly rmsd cutoff value in nm.

        num_clusters : int , optional
            Number of clusters. Default = ss_optimal_num_clusters
        
        """ 
        

        if any(k not in self.has_been_called for k in ('assembly_rmsd' , 'pose_rmsd' , 'aligned_pos')) :

            print("Please call assembly_rmsd(), pose_rmsd(), and aligned_pos() methods before calling clustering_multi_hdf5()")
            return None
            
        else:
            
            
    

            # if num_clusters is None:
            #     num_clusters = self.optimal_num_clusters

            self.has_been_called["clustering_multi_hdf5"] = True
            self.assembly_rmsds_array = assembly_rmsds_array
            self.pose_rmsds_array = pose_rmsds_array
            self.aligned_poses_array = aligned_poses_array
                
            valid_frames = self.assembly_rmsds_array < assembly_rmsd_cutoff 

            self.valid_aligned_pos = self.aligned_poses_array[valid_frames]
            self.valid_walker_idxs , self.valid_cycle_idxs = np.where(valid_frames)

            num_frames = np.shape(self.valid_aligned_pos)[0]
            num_clusters = int(num_frames / 10)
            self.num_atoms = np.shape(self.valid_aligned_pos)[1]
            valid_aligned_pos_reshaped = self.valid_aligned_pos.reshape(num_frames, self.num_atoms * 3)
            
            # find the optimal number of clusters
            kmeans = KMeans(n_clusters=num_clusters, n_init = 10, random_state=42).fit(valid_aligned_pos_reshaped)
            cluster_centers = (kmeans.cluster_centers_).reshape(num_clusters,self.num_atoms,3)
            labels = kmeans.labels_

            # obtain the most populated cluster center                                                                               
            population = []
            for i in np.unique(labels):
                pop = len(np.where(labels == i)[0])
                population.append(pop)


            most_pop_cluster_index = np.argmax(population)
            self.most_pop_cluster_center = cluster_centers[most_pop_cluster_index].reshape(self.num_atoms,3)
            self.frame_idxs = np.where(labels == most_pop_cluster_index)[0]

            return self.most_pop_cluster_center , self.frame_idxs

    def find_centroid_multi_hdf5(self, multi_hdf5_output_dir, run_nums, print_results = True ): 
        """Calculate the centroid ghost particles positions among the valid farmes of multiple runs with the same number of walkers.
        Return the centroid summary, and save the centroid pdb.

        parameters
        ----------
        multi_hdf5_output_dir : string
            path to the output directory for multi-hdf5 analysis
            
        run_nums : list of int
            list of the run numbers.

        print_results : bool, defult = True
 
        """
        if ('clustering_multi_hdf5' not in self.has_been_called ) :

            print("Please call clustering_multi_hdf5() method before calling find_centroid_multi_hdf5().")
            return None
            
        else:
            self.has_been_called["find_centroid_multi_hdf5"] = True
            self.multi_hdf5_output_dir = multi_hdf5_output_dir
            self.run_nums = run_nums

            distances = ave_distances  = np.zeros(len(self.frame_idxs))
            
            for i,v in enumerate(self.frame_idxs):
                for j in range(self.num_atoms):
                    distances [i] += distance.euclidean(self.most_pop_cluster_center[j], self.valid_aligned_pos[v][j])
                ave_distances [i] = distances [i] / self.num_atoms

            
            min_ave_distance = np.amin(ave_distances)
            centroid_frame_index = self.frame_idxs[np.where(ave_distances == min_ave_distance)]
            centroid_pos = self.valid_aligned_pos[centroid_frame_index].reshape(self.num_atoms,3)
            centroid_walker_index = self.valid_walker_idxs[centroid_frame_index][0]
            centroid_cycle_index = self.valid_cycle_idxs[centroid_frame_index][0]
            centroid_assembly_rmsd =  np.array(self.assembly_rmsds_array)[centroid_walker_index , centroid_cycle_index]
            centroid_pose_rmsd = np.array(self.pose_rmsds_array)[centroid_walker_index , centroid_cycle_index]
            centroid_run_index, centroid_final_walker_index = divmod(centroid_walker_index, self.walkers)
            centroid_final_run_index = self.run_nums[centroid_run_index]  
            
            self.centroid_summary_multi_hdf5 = {'title': "centroid_summary_multi_hdf5",
                                'min_ave_distance' : min_ave_distance,
                                'run' : centroid_final_run_index,
                                'walker' : centroid_final_walker_index,
                                'cycle' : centroid_cycle_index,
                                'assembly_rmsd': centroid_assembly_rmsd,
                                'pose_rmsd' : centroid_pose_rmsd}

            # save the pdb of centroid                                                                                              
            centered_lig_idxs = self.ref_pdb.top.select("chainid 1")
            centroid_pdb = mdj.Trajectory(centroid_pos , self.ref_pdb.atom_slice(centered_lig_idxs).top)
            os.makedirs(self.multi_hdf5_output_dir , exist_ok=True)
            centroid_pdb_path = osp.join(self.multi_hdf5_output_dir,f'centroid_pdb_multi_hdf5.pdb' )
            centroid_pdb.save_pdb(centroid_pdb_path)
         
            if (print_results):
                print("Centroid Summary:")
                for key, value in self.centroid_summary_multi_hdf5.items():
                    print(key, ':' , value)
                       
            return self.centroid_summary_multi_hdf5
      
    
    
    def plot_multi_hdf5 (self, run_nums, multi_hdf5_output_dir, assembly_rmsds_array, pose_rmsds_array, losses_array):
        """Plot assembly and pose RMSD and loss per particles graphs for multi-hdf5 analysis"""


        if any(k not in self.has_been_called for k in ('assembly_rmsd' , 'pose_rmsd' , 'calculate_loss')) :

            print("Please call assembly_rmsd(), pose_rmsd(), and calculate_loss_per_particle methods before calling plot()")
            return None    

        else:
            self.run_nums = run_nums
            self.multi_hdf5_output_dir = multi_hdf5_output_dir
            self.assembly_rmsds_array = assembly_rmsds_array
            self.pose_rmsds_array = pose_rmsds_array
            self.losses_array = losses_array
            self.walkers_multi_hdf5 = assembly_rmsds_array.shape[0]
            

            plot_path = osp.join(self.multi_hdf5_output_dir,'plots')
            os.makedirs(plot_path, exist_ok=True)

            cycles_array= range(0,self.cycles,1)
            ave_pose_rmsd = np.mean(self.pose_rmsds_array,axis=0)
            ave_assembly_rmsd = np.mean(self.assembly_rmsds_array,axis=0)

            assemblyRMSD_fig = plt.figure(figsize = (5,5))
            for walker in range (self.walkers_multi_hdf5):
                plt.plot(cycles_array,self.assembly_rmsds_array[walker],color='gray',ls='-',lw=3 , alpha = 0.3)
            plt.plot(cycles_array,ave_assembly_rmsd,color='black',ls='-',lw=3 , label = 'Assembly RMSD')
            
            plt.xlabel("Cycle",fontsize=12,fontweight='bold',color='k')
            plt.ylabel("Assembly RMSD (nm)",fontsize=12,fontweight='bold',color='k')
            plt.xticks(np.arange(0,100.1 , step = 20))
            plt.yticks(np.arange(0,0.51 , step = 0.1))
            plt.legend()
            save_fig(f'assembly_rmsd' , plot_path )
            plt.show
            

            poseRMSD_fig = plt.figure(figsize = (5,5))
            for walker in range (self.walkers_multi_hdf5):
                plt.plot(cycles_array,self.pose_rmsds_array[walker],color='gray',ls='-',lw=3 ,  alpha = 0.3)
            plt.plot(cycles_array,ave_pose_rmsd,color='black',ls='-',lw=3 , label = 'Pose RMSD')

            plt.xlabel("Cycle",fontsize=12,fontweight='bold',color='k')
            plt.ylabel("Pose RMSD (nm)",fontsize=12,fontweight='bold',color='k')
            plt.xticks(np.arange(0,100.1 , step = 20))
            plt.yticks(np.arange(0,0.71 , step = 0.1))
            plt.legend()
            save_fig(f'pose_rmsd' , plot_path )
            plt.show

            ave_loss = np.mean(self.losses_array,axis=0)
        

            loss_fig = plt.figure(figsize = (5,5))
            for walker in range (self.walkers_multi_hdf5):
                plt.plot(cycles_array,self.losses_array[walker],color='gray',ls='-',lw=3 ,  alpha = 0.3)
            plt.plot(cycles_array,ave_loss,color='black',ls='-',lw=3 , label = 'Loss per Particle')

            plt.xlabel("Cycle",fontsize=12,fontweight='bold',color='k')
            plt.ylabel("Loss per Particle",fontsize=12,fontweight='bold',color='k')
            plt.legend()
            save_fig(f'loss' , plot_path )
            plt.show

            loss_RMSD_fig = plt.figure(figsize = (5,5))
            for walker in range (self.walkers_multi_hdf5):
                plt.plot(self.assembly_rmsds_array[walker],self.losses_array[walker], 'o' , color='gray', alpha = 0.3)
            
            plt.xlabel("RMSD",fontsize=12,fontweight='bold',color='k')
            plt.ylabel("Loss per Particle",fontsize=12,fontweight='bold',color='k')
            plt.yscale('log')
            save_fig(f'loss_RMSD' , plot_path )
            plt.show


    def save_summary_multi_hdf5(self):
        """save the summary of the rmsd analysis and centroid in a txt file.
        
        """

        if any(k not in self.has_been_called for k in ('rmsd_analysis_multi_hdf5' , 'find_centroid_multi_hdf5')) :

            print("Please call rmsd_analysis_multi_hdf5(), and find_centroid_multi_hdf5() methods before calling save_summary_multi_hdf5()")
            return None 

        else:   
            
            with open(osp.join(self.multi_hdf5_output_dir , "summary.txt") , 'w') as file:
                
                for key, value in self.rmsd_analysis_summary_multi_hdf5.items():
                    line = f"{key}: {value}\n"
                    file.write(line)
                for key, value in self.centroid_summary_multi_hdf5.items():
                    line = f"{key}: {value}\n"
                    file.write(line)

    def convert_hdf5_to_dcd_attr(self , minimized_pdb_path , plot_attrs=False , walker_num_to_plot=None):
        """Convert the wepy_HDF5 filr to dcds and extract attributes and save that as a text file.

        parameters
        ----------
        minimized_pdb_path : string
            path to the minimized_pos_pdb pdb file.

        plot_attrs : bool, defult = False
            
        walker_num_to_plot : list of int , default = None
            list of the walker numbers for plotting the attributes.
 
        """
        
        self.minimized_pdb_path = minimized_pdb_path
        self.plot_attrs = plot_attrs
        self.walker_num_to_plot = walker_num_to_plot
        
        dcd_attrs_path = osp.join(self.base_dir , f'dcd_attrs')
        os.makedirs(dcd_attrs_path, exist_ok=True)
        pdb = mdj.load_pdb(minimized_pdb_path)
        top = pdb.topology
        
        # Make sure the number of atoms matched in the pdb and the h5
        pdb_atoms = len(pdb.xyz[0]) 
        if self.num_atoms == pdb_atoms:
            print('The number of atoms in the pdb and h5 matches.')
            
        print(f'This H5 has {self.runs} run(s), {self.walkers} walkers, and {self.cycles} cycles.')

        
        print('Making no resampling DCDs.')

        with self._h5:
            for i in range (self.walkers)  :
                pos = np.array(self._h5.h5[f'runs/0/trajectories/{i}/positions'])
                box_vectors = np.array(self._h5.h5[f'runs/0/trajectories/{i}/box_vectors'])
                corrected_bvs = []
                
                for j,v in enumerate(box_vectors):
                
                    tmp = np.mean(v, axis=0)
                    corrected_bvs.append(np.ndarray.tolist(tmp))

                unitcell_angles = np.zeros(shape=(self.cycles,3)) #shape is (n_cycles, 3)
                unitcell_angles.fill(90)

                traj = mdj.Trajectory(pos, top,
                                    unitcell_lengths=corrected_bvs,
                                    unitcell_angles=unitcell_angles)
                traj.save_dcd(dcd_attrs_path +'/'+'nores_walker'+str(i)+'.dcd')
                print('DCD file of walker {} is saved'.format(i))
                
                # Save all attributes for all walkers
                charges , sigmas , epsilons, alambdas = [] , [] , [] , [] 
                for i in range(self.walkers):
                    charge = np.array(self._h5.h5[f'/runs/0/trajectories/{i}/charges']).reshape(-1, 1)
                    sigma = np.array(self._h5.h5[f'/runs/0/trajectories/{i}/sigmas']).reshape(-1, 1)
                    epsilon = np.array(self._h5.h5[f'/runs/0/trajectories/{i}/epsilons']).reshape(-1, 1)
                    alambda = np.array(self._h5.h5[f'/runs/0/trajectories/{i}/lambdas']).reshape(-1, 1)
                    charges.append(charge)
                    sigmas.append(sigma)
                    epsilons.append(epsilon)
                    alambdas.append(alambda)
                    attributes = np.hstack((charge, sigma, epsilon, alambda))
                    np.savetxt(osp.join(dcd_attrs_path,f'all_attr{i}.txt'), attributes, delimiter=' ')  

        print("All Attributes Are Saved Succesfully")

        if self.walker_num_to_plot:
            if walker_num_to_plot == None:
                print("Set the list of walker idxs to plot the attributes!")

            else:
                
                num_gho = int(alambda.shape[0] / self.cycles)
                cycles_array= range(0,self.cycles,1)
                
                for i,v in enumerate(self.walker_num_to_plot):
                    a = alambdas[v].reshape(self.cycles, num_gho)
                    c = charges[v].reshape(self.cycles, num_gho)
                    e = epsilons[v].reshape(self.cycles, num_gho)
                    s = sigmas[v].reshape(self.cycles, num_gho)
                    
                    fig, ax = plt.subplots(4, 1)
                    fig.suptitle(f'Attributes of walker{v}')
                    ax[0].plot(cycles_array , a)
                    ax[1].plot(cycles_array , c)
                    ax[2].plot(cycles_array , s)
                    ax[3].plot(cycles_array , e)

                    ax[3].set_xlabel("Cycle", fontsize=12,fontweight='bold',color='k')
                    plt.setp(ax[0].get_xticklabels(), visible=False)
                    plt.setp(ax[1].get_xticklabels(), visible=False)
                    plt.setp(ax[2].get_xticklabels(), visible=False)
                    ax[0].set_ylabel("Lambda" , fontsize=12,fontweight='bold',color='k')
                    ax[1].set_ylabel("Charge" , fontsize=12,fontweight='bold',color='k')
                    ax[2].set_ylabel("Sigma", fontsize=12,fontweight='bold',color='k')
                    ax[3].set_ylabel("Epsilon", fontsize=12,fontweight='bold',color='k')
    
                    attr_plot_path = osp.join(self.base_dir , "plots" , "attrs")
                    os.makedirs(attr_plot_path , exist_ok=True)

                    save_fig(f'attributes_walker{v}' , attr_plot_path )
                    plt.show
                