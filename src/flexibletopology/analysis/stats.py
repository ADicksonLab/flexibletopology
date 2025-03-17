import numpy as np

def GetVolumeAndPSA(state,bin_res=0.1,polar_cutoff=0.3):
    """
    Uses a grid to determine the volume of a set of ghost particles.
    """

    max_sigma = state['sigma'].max()
    bin_minima = np.min(state['positions'],axis=0) - max_sigma
    maxs = np.max(state['positions'],axis=0) + max_sigma

    # get number of bins
    num_bins = np.array(np.floor((maxs - bin_minima)/bin_res) + 1,dtype=int)

    # get a list of bin indices for each particle
    n_ghosts = state['positions'].shape[0]
    rad_sq = np.square(0.5612310*state['sigma'])  # 0.5 * 2^1/6 = 0.5612310
    bin_occ = np.zeros(num_bins,dtype=bool)
    polar_bins = []
    all_bin_idxs = []
    for i in range(n_ghosts):
        # find center bin
        center_bin = np.array((state['positions'][i] - bin_minima)/bin_res,dtype=int)
            
        # how many bins to examine in each direction?
        bin_delta = int(state['sigma'][i]/bin_res) + 1

        # loop over everything in a cube and keep the ones whose distance to the particle is less than sigma/2
        for x_idx in range(center_bin[0]-bin_delta,center_bin[0]+bin_delta+1):
            if x_idx >= 0 and x_idx < num_bins[0]:
                x = bin_minima[0] + x_idx*bin_res
                for y_idx in range(center_bin[1]-bin_delta,center_bin[1]+bin_delta+1):
                    if y_idx >= 0 and y_idx < num_bins[1]:
                        y = bin_minima[1] + y_idx*bin_res
                        for z_idx in range(center_bin[2]-bin_delta,center_bin[2]+bin_delta+1):
                            if z_idx >= 0 and z_idx < num_bins[2]:
                                z = bin_minima[2] + z_idx*bin_res
                                bin_pos = np.array([x,y,z])
                                if np.sum(np.square(bin_pos - state['positions'][i])) < rad_sq[i]:
                                    bin_occ[x_idx,y_idx,z_idx] = True
                                    if np.abs(state['charge'][i]) > polar_cutoff:
                                        if [x_idx,y_idx,z_idx] not in polar_bins:
                                            polar_bins.append([x_idx,y_idx,z_idx])
                                    
                                    bin_idx = x_idx*num_bins[1]*num_bins[2] + y_idx*num_bins[2] + z_idx
                                    all_bin_idxs.append(bin_idx)

    # count the number of unique bins occupied
    unique_bins = len(set(all_bin_idxs))

    # for each polar bin, count the number of faces that are unoccupied
    nbor_count = 0
    for p in polar_bins:
        if p[0] - 1 < 0:
            nbor_count += 1
        elif not bin_occ[p[0]-1,p[1],p[2]]:
            nbor_count += 1

        if p[0] + 1 >= num_bins[0]:
            nbor_count += 1
        elif not bin_occ[p[0]+1,p[1],p[2]]:
            nbor_count += 1

        if p[1] - 1 < 0:
            nbor_count += 1
        elif not bin_occ[p[0],p[1]-1,p[2]]:
            nbor_count += 1

        if p[1] + 1 >= num_bins[1]:
            nbor_count += 1
        elif not bin_occ[p[0],p[1]+1,p[2]]:
            nbor_count += 1

        if p[2] - 1 < 0:
            nbor_count += 1
        elif not bin_occ[p[0],p[1],p[2]-1]:
            nbor_count += 1

        if p[2] + 1 >= num_bins[2]:
            nbor_count += 1
        elif not bin_occ[p[0],p[1],p[2]+1]:
            nbor_count += 1

    bin_vol = bin_res**3
    bin_face_area = bin_res**2
            
    return unique_bins * bin_vol, nbor_count*bin_face_area

def GetDipole(state):
    charge_vec = np.zeros((3))
    com = np.mean(state['positions'],axis=0)

    for i in range(len(state['positions'])):
        charge_vec += state['charge'][i]*(state['positions'][i] - com)

    return charge_vec

