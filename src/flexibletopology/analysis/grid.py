import numpy as np

PI_FAC = 4.0*np.pi/3.0

class Grid(object):
    """
    A 3D lattice used for analysis of flexible topology simulations.

    Input:
    bin_res : (float) The resolution of the grid

    state_list (optional): (list of dict) Each dict contains FT states including 
                           particle positions which are an array of shape (N,3)

    particle_buffer : (float) Spacing added to the maximum and minimum coordinate values.
                      (default 0.2 nm)

    xyz_limits (optional): (array with shape (3,2)) Max and min values for the grid
                           along x,y,z dimensions.
    """
    def __init__(self, bin_res, state_list=None, particle_buffer=0.2, xyz_limits=None):
        self.bin_res = bin_res
        if state_list is not None:
            xyz_set = np.concatenate([s['positions'] for s in state_list],axis=0)
            self.bin_minima = np.min(xyz_set,axis=0) - particle_buffer
            maxs = np.max(xyz_set,axis=0) + particle_buffer
        else:
            assert xyz_limits is not None, "Must define either xyz_limits or xyz_set to define a Grid"
            self.bin_minima = xyz_limits[:,0]
            maxs = xyz_limits[:,1]

        # get number of bins                                                                                            
        self.num_bins = np.array(np.floor((maxs - self.bin_minima)/self.bin_res) + 1,dtype=int)

    def get_single_xyz(self,i,j,k):
        """
        Get the x,y,z coordinates corresponding to a set of grid indices
        """
        x = self.bin_minima[0] + i*self.bin_res
        y = self.bin_minima[1] + j*self.bin_res
        z = self.bin_minima[2] + k*self.bin_res
        return np.array(x,y,z)

    def get_xyz(self):
        """
        Return a big numpy array that holds all of the xyz coordinates for each bin.
        """
        big_arr = np.zeros((self.num_bins[0],self.num_bins[1],self.num_bins[2],3))
        for i in range(self.num_bins[0]):
            x = self.bin_minima[0] + i*self.bin_res
            big_arr[i,:,:,0] = x

        for j in range(self.num_bins[1]):
            y = self.bin_minima[1] + j*self.bin_res
            big_arr[:,j,:,1] = y

        for k in range(self.num_bins[2]):
            z = self.bin_minima[2] + k*self.bin_res
            big_arr[:,:,k,2] = z

        return big_arr

            
    def get_bin(self,pos):
        """
        Returns the bin indices corresponding to a set of xyz coordinates.
        """
        return np.array((pos - self.bin_minima)/self.bin_res,dtype=int)

    def get_xyz_subarray(self,min_idxs,max_idxs):
        """
        Gets an array of grid positions for a specified range of indices.
        """
        subarray = np.zeros((max_idxs[0]-min_idxs[0]+1,
                             max_idxs[1]-min_idxs[1]+1,
                             max_idxs[2]-min_idxs[2]+1,
                             3))
        for i in range(min_idxs[0],max_idxs[0]+1):
            x = self.bin_minima[0] + i*self.bin_res
            for j in range(min_idxs[1],max_idxs[1]+1):
                y = self.bin_minima[1] + j*self.bin_res
                for k in range(min_idxs[2],max_idxs[2]+1):
                    z = self.bin_minima[2] + k*self.bin_res
                    subarray[i-min_idxs[0],j-min_idxs[1],k-min_idxs[2]] = x,y,z

        return subarray

    def stamp_particle(self,pos,radius):
        """
        Returns a list of bin indices that touch a particle centered at pos
        with a given radius.
        """
        radius_sq = radius*radius
        
        center_bin = self.get_bin(pos)

        n_bins_around_center = int(radius/self.bin_res) + 1
        min_idxs = center_bin - n_bins_around_center
        max_idxs = center_bin + n_bins_around_center

        # get an array of bin coordinates
        bins_test_xyz = self.get_xyz_subarray(min_idxs,max_idxs)

        # get distance of each bin center to pos
        dists = np.square(bins_test_xyz - pos).sum(axis=3)

        idx_tup = np.where(dists < radius_sq)

        bin_list = [[i+min_idxs[0],j+min_idxs[1],k+min_idxs[2]] for i,j,k in zip(*idx_tup)]
        bin_list.append([center_bin[0],center_bin[1],center_bin[2]])

        return bin_list

    def compute_elec_pot_grid(self,state):
        # get array of grid positions
        grid_arr = self.get_xyz()

        n_ghosts = state['positions'].shape[0]

        e_grid = np.zeros((grid_arr.shape[0],grid_arr.shape[1],grid_arr.shape[2]))
        masks = []
        for i in range(n_ghosts):
            d_grid = np.sqrt(np.square(grid_arr-state['positions'][i]).sum(axis=3))
            masks.append(d_grid < 0.5*state['sigma'][i])
            e_grid += np.divide(state['charge'][i], d_grid, out=np.zeros_like(d_grid), where=d_grid!=0)
            
        # use masks to set interior elec_potential to zero
        combined_mask = np.logical_or.reduce(masks)
        e_grid[combined_mask] = 0

        return e_grid

    def compute_signed_elec_pot_grid(self,state,cutoff=0.2):
        # get array of grid positions
        grid_arr = self.get_xyz()

        n_ghosts = state['positions'].shape[0]

        e_grid = np.zeros((grid_arr.shape[0],grid_arr.shape[1],grid_arr.shape[2]))
        masks = []
        for i in range(n_ghosts):
            d_grid = np.sqrt(np.square(grid_arr-state['positions'][i]).sum(axis=3))
            masks.append(d_grid < 0.5*state['sigma'][i])
            e_grid += np.divide(state['charge'][i], d_grid, out=np.zeros_like(d_grid), where=d_grid!=0)
            
        # use masks to set interior elec_potential to zero
        combined_mask = np.logical_or.reduce(masks)
        e_grid[combined_mask] = 0

        e_grid[np.abs(e_grid) < cutoff] = 0
        e_grid[e_grid > cutoff] = 1
        e_grid[e_grid < -cutoff] = -1
        
        return e_grid
    
    def compute_occ(self,state,max_occ=1):
        # determine a list of bin indices for each particle
        n_ghosts = state['positions'].shape[0]
        all_bins = []
        for i in range(n_ghosts):
            bin_list = self.stamp_particle(state['positions'][i],state['sigma'][i]*0.5)
            all_bins.append(bin_list)

        # use lambdas to add particles to occupancy array at the appropriate bin indices
        occ = np.zeros((self.num_bins[0],self.num_bins[1],self.num_bins[2]))
        for at_idx, blist in enumerate(all_bins):
            for b in blist:
                occ[b[0],b[1],b[2]] += state['lambda'][at_idx]

        occ = np.clip(occ,0,max_occ)

        return occ
    
    def compute_vol(self,state):
        occ = self.compute_occ(state)
        unique_bins = occ.sum()
        bin_vol = self.bin_res**3
        
        return unique_bins*bin_vol

    def compute_dmax(self,state):
        n = len(state['positions'])
        dmax = 0
        for i in range(n-1):
            for j in range(i+1,n):
                d = np.sqrt(np.square(state['positions'][i] - state['positions'][j]).sum())
                d += 0.5*(state['sigma'][i] + state['sigma'][j])
                if d > dmax:
                    dmax = d
        return dmax

    def compute_globularity(self, state):
        """
        Returns the ratio of the ligand volume over the volume of the minimum encompassing sphere 
        (approximated as 4/3 * pi * R_max**3, where R_max is 0.5 times the largest particle-particle distance)
        """

        max_d = self.compute_dmax(state)
        vol = self.compute_vol(state)

        sphere_vol = PI_FAC*(0.5*max_d)**3
        return vol/sphere_vol
    
    def print_pos_neg_image(self, image, flabel):
        """ 
        Saves two Gaussian-style .cube file of 'image'
        at each point in the grid, one for positive values
        and one for negative values.
        """
        NM_TO_BOHRS = 18.8973
        natoms = 1
        cx,cy,cz = 0.0,0.0,0.0
        fname1 = f'{flabel}_pos.cube'
        fname2 = f'{flabel}_neg.cube'
        f1 = open(fname1,'w')
        f2 = open(fname2,'w')

        print("Positive electrostatic density",file=f1)
        print("Created by Flexible Topology.",file=f1)
        print("Negative electrostatic density",file=f2)
        print("Created by Flexible Topology.",file=f2)
        
        print(f"{natoms:4d} {self.bin_minima[0]*NM_TO_BOHRS:11.6f} {self.bin_minima[1]*NM_TO_BOHRS:11.6f} {self.bin_minima[2]*NM_TO_BOHRS:11.6f}",file=f1)
        print(f"{natoms:4d} {self.bin_minima[0]*NM_TO_BOHRS:11.6f} {self.bin_minima[1]*NM_TO_BOHRS:11.6f} {self.bin_minima[2]*NM_TO_BOHRS:11.6f}",file=f2)
        
        print(f"{self.num_bins[0]:4d} {self.bin_res*NM_TO_BOHRS:11.6f} {0.0:11.6f} {0.0:11.6f}",file=f1)
        print(f"{self.num_bins[1]:4d} {0.0:11.6f} {self.bin_res*NM_TO_BOHRS:11.6f} {0.0:11.6f}",file=f1)
        print(f"{self.num_bins[2]:4d} {0.0:11.6f} {0.0:11.6f} {self.bin_res*NM_TO_BOHRS:11.6f}",file=f1)
        
        print(f"{self.num_bins[0]:4d} {self.bin_res*NM_TO_BOHRS:11.6f} {0.0:11.6f} {0.0:11.6f}",file=f2)
        print(f"{self.num_bins[1]:4d} {0.0:11.6f} {self.bin_res*NM_TO_BOHRS:11.6f} {0.0:11.6f}",file=f2)
        print(f"{self.num_bins[2]:4d} {0.0:11.6f} {0.0:11.6f} {self.bin_res*NM_TO_BOHRS:11.6f}",file=f2)
        
        print(f"{1:4d} {0.0:11.6f} {cx:11.6f} {cy:11.6f} {cz:11.6f}",file=f1)
        print(f"{1:4d} {0.0:11.6f} {cx:11.6f} {cy:11.6f} {cz:11.6f}",file=f2)
        
        for i in range(self.num_bins[0]):
            for j in range(self.num_bins[1]):
                toprint1 = ""
                toprint2 = ""
                for k in range(self.num_bins[2]):
                    if image[i][j][k] > 0:
                        toprint1 += f'{image[i][j][k]:12.5E}  '
                        toprint2 += f'{0:12.5E}  '
                    else:
                        toprint2 += f'{-image[i][j][k]:12.5E}  '
                        toprint1 += f'{0:12.5E}  '

                    if k % 6 == 5:
                        print(toprint1,file=f1)
                        print(toprint2,file=f2)
                        toprint1 = ""
                        toprint2 = ""

                if len(toprint1) > 0:
                    print(toprint1,file=f1)
                if len(toprint2) > 0:
                    print(toprint2,file=f2)

        f1.close()
        f2.close()
        return

    def unflatten(self,array):
        return array.reshape(self.num_bins[0],self.num_bins[1],self.num_bins[2])
