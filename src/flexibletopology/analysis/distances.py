import numpy as np
from flexibletopology.analysis.grid import Grid


CHARGE_IDX = 0
SIGMA_IDX = 1
EPSILON_IDX = 2
LAMBDA_IDX = 3    

class FTDistance(object):
    """
    A base class for state-to-state distance metrics for flexible topology systems.
    """

    def image(self,state):
        """
        Reduces a flexible topology state to the minimum information required to calculate the distance.
        
        The state is a dictionary that contains 'positions', 'charge', 'sigma', 'epsilon' and 'lambda' as keys.
        """
        raise NotImplementedError

    def image_distance(self,image1,image2):
        """
        Computes the distance between the images.
        """
        raise NotImplementedError

    def _calibrate(self,state_list):
        pass
    
    def all_to_all(self,state_list,return_images=False):
        self._calibrate(state_list)
        
        images = [self.image(state) for state in state_list]

        n = len(images)
        dists = np.zeros((n,n))
        for i in range(n-1):
            for j in range(i+1,n):
                dists[i,j] = self.image_distance(images[i],images[j])
                dists[j,i] = dists[i,j]

        if return_images:
            return dists, images
        else:
            return dists

    def list_to_list(self,state_list1,state_list2,return_images=False):
        self._calibrate(state_list1 + state_list2)
        
        images1 = [self.image(state) for state in state_list1]
        images2 = [self.image(state) for state in state_list2]

        n1 = len(images1)
        n2 = len(images2)

        dist1to2 = np.zeros((n1,n2))
        
        for i in range(n1):
            for j in range(n2):
                dist1to2[i,j] = self.image_distance(images1[i],images2[j])

        if return_images:
            return dist1to2, images1, images2
        else:
            return dist1to2

class GridDistance(FTDistance):
    """
    An extended abstract base class that includes defining a Grid object 
    upon calibration.
    """
    def __init__(self,particle_buffer=0.2):
        self.calibrated = False
        self.particle_buffer = particle_buffer
        

    def _calibrate(self,state_list):
        self.grid = Grid(self.bin_res, state_list, particle_buffer=self.particle_buffer)
        self.calibrated = True

class GeoFTDistance(GridDistance):
    """
    Uses a grid to compare FT systems that may or may not have the same number of particles.
    """

    def __init__(self, bin_res=0.1, max_occ=1):
        self.bin_res = bin_res
        self.max_occ = max_occ

    def _binify_particles(self,xyz,sigmas):
        if not self.calibrated:
            raise ValueError('GeoFTDistance is uncalibrated when calling _binify_particles!')

        # returns a list of bin indices for each particle
        
        n_ghosts = xyz.shape[0]
        all_bins = []
        for i in range(n_ghosts):
            bin_list = self.grid.stamp_particle(xyz[i],sigmas[i]*0.5)
            all_bins.append(bin_list)

        return all_bins
                 
    def image(self,state):
        bin_lists = self._binify_particles(state['positions'],state['sigma'])

        occ = np.zeros((self.grid.num_bins[0],self.grid.num_bins[1],self.grid.num_bins[2]))
        for at_idx, blist in enumerate(bin_lists):
            for b in blist:
                occ[b[0],b[1],b[2]] += state['lambda'][at_idx]

        occ = np.clip(occ,0,self.max_occ)
        grid = {'occupancy' : occ,
                'xyz_mins' : self.grid.bin_minima,
                'n_xyz' : self.grid.num_bins,
                'bin_res' : self.bin_res}

        return grid

    def image_distance(self,image1,image2):
        frac_non_overlap = np.sum(np.abs(image1['occupancy']-image2['occupancy']))/(np.sum(image1['occupancy']) + np.sum(image2['occupancy']))
        return frac_non_overlap
        
class ChargeGeoFTDistance(GeoFTDistance):
    """
    Same as GeoFTD but using a separate grid for neutral, positive and negative atoms.
    """
    def __init__(self, bin_res=0.1, max_occ=1, neg_cutoff=-0.3, pos_cutoff=0.3):

        super().__init__(bin_res=bin_res, max_occ=max_occ)
        self.neg_cutoff = neg_cutoff
        self.pos_cutoff = pos_cutoff
         
    def _classify_particles(self, charges):
        neg_list = []
        neut_list = []
        pos_list = []
        for idx, c in enumerate(charges):
            if c < self.neg_cutoff:
                neg_list.append(idx)
            elif c > self.pos_cutoff:
                pos_list.append(idx)
            else:
                neut_list.append(idx)
        #print(len(neg_list),len(neut_list),len(pos_list))
        return neg_list, neut_list, pos_list
                

    def image(self,state):
        neg_list, neut_list, pos_list = self._classify_particles(state['charge'])

        bin_lists = []
        bin_lists.append(self._binify_particles(state['positions'][neg_list],state['sigma'][neg_list]))
        bin_lists.append(self._binify_particles(state['positions'][neut_list],state['sigma'][neut_list]))
        bin_lists.append(self._binify_particles(state['positions'][pos_list],state['sigma'][pos_list]))

        occ = np.zeros((3,self.grid.num_bins[0],self.grid.num_bins[1],self.grid.num_bins[2]))

        for bin_idx in range(len(bin_lists)):
            for at_idx, blist in enumerate(bin_lists[bin_idx]):
                for b in blist:
                    occ[bin_idx,b[0],b[1],b[2]] += state['lambda'][at_idx]

        occ = np.clip(occ,0,self.max_occ)

        grid = {'occupancy' : occ,
                'xyz_mins' : self.grid.bin_minima,
                'n_xyz' : self.grid.num_bins,
                'bin_res' : self.bin_res}

        return grid

class ChargeDensityGridDistance(GeoFTDistance):
    """
    Same as GeoFTD but sums the charges instead of the densities.
    """
    def __init__(self, bin_res=0.3, max_abs_charge=0.4):

        super().__init__(bin_res=bin_res)
        self.max_abs_charge = max_abs_charge
    
    def image(self,state):
        bin_lists = self._binify_particles(state['positions'],state['sigma'])
        bin_charge = np.zeros((self.grid.num_bins[0],self.grid.num_bins[1],self.grid.num_bins[2]))

        for at_idx, blist in enumerate(bin_lists):
            for b in blist:
                bin_charge[b[0],b[1],b[2]] += state['lambda'][at_idx]*state['charge'][at_idx]

        bin_charge = np.clip(bin_charge,-self.max_abs_charge,self.max_abs_charge)

        grid = {'occupancy' : bin_charge,
                'xyz_mins' : self.grid.bin_minima,
                'n_xyz' : self.grid.num_bins,
                'bin_res' : self.bin_res}

        return grid

    def image_distance(self,image1,image2):
        im1 = image1['occupancy']
        im2 = image2['occupancy']
        frac_non_overlap = np.sum(np.abs(im1-im2))/(np.sum(np.abs(im1)) + np.sum(np.abs(im2)))
        return frac_non_overlap

    
class ElecGridDistance(GridDistance):
    """
    Uses a grid to compare FT systems that may or may not have the same number of particles.
    """

    def __init__(self, bin_res=0.1, particle_buffer=0.4):
        super().__init__(particle_buffer=particle_buffer)
        self.bin_res = bin_res

    def image(self,state):

        e_grid = self.grid.compute_elec_pot_grid(state)
        
        grid = {'occupancy' : e_grid,
                'xyz_mins' : self.grid.bin_minima,
                'n_xyz' : self.grid.num_bins,
                'bin_res' : self.bin_res}

        return grid

    def image_distance(self,image1,image2):
        im1 = image1['occupancy']
        im2 = image2['occupancy']
        frac_non_overlap = np.sum(np.abs(im1-im2))/(np.sum(np.abs(im1)) + np.sum(np.abs(im2)))

        return frac_non_overlap

class ElecFTDistance(FTDistance):
    """
    Gets the ghost-particle-induced electric potential (in units of k), evaluated at key points in the space, and 
    uses that to measure the distance between frames.
    """

    def __init__(self, xyz_list):
        self.xyz_list = xyz_list

    def image(self,state):
        n_points = len(self.xyz_list)
        elec_pot = np.zeros((n_points))
        for pt_idx, pos in enumerate(self.xyz_list):
            for gh_idx, gh_pos in enumerate(state['positions']):
                r = np.sqrt(np.sum(np.square(pos-gh_pos)))
                elec_pot[pt_idx] += state['charge'][gh_idx]/r;

        # divide by variance for better comparison
        var = np.sqrt(np.sum(np.square(elec_pot)))
        return elec_pot/var


    def image_distance(self,image1,image2):
        return np.sum(np.square(image1-image2))

