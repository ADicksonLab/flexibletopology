import numpy as np

def gen_init_attr(n_ghosts, attr_bounds,total_charge=None,init_lambda=None):

    initial_attr = {}
        
    for k in attr_bounds.keys():
        initial_attr[k] = np.random.uniform(low=attr_bounds[k][0], high=attr_bounds[k][1], size=(n_ghosts))

    if total_charge is not None:
        # set total charge to fixed value
        total_charge = initial_attr['charge'].sum()
        initial_attr['charge'] -= total_charge/n_ghosts

    if init_lambda is not None:
        initial_attr['lambda'][:] = init_lambda
    
    return initial_attr


def gen_init_pos(n_ghosts, COM_BS, WIDTH, pdb_pos, min_dist=0.1, gg_min_dist=0.1, gg_max_dist=0.25, first_width=0.3, max_trials=10000, max_attempts=10):

    for n_attempts in range(max_attempts):
        n_trials_this_attempt = 0
        rand_positions = []
        
        while len(rand_positions) < 1:
            if n_trials_this_attempt > max_trials:
                break
            n_trials_this_attempt += 1
            r_pos = np.random.uniform(low=-first_width, high=first_width,size=(1, 3))
            r_pos = r_pos+COM_BS
            if len(pdb_pos) > 0:
                # check if this ghost particle is too close to any system atoms
                dists = np.linalg.norm(pdb_pos - r_pos, axis=1)
                if np.all(dists > min_dist):
                    rand_positions.append(r_pos)
            else:
                rand_positions.append(r_pos)
                
        while len(rand_positions) < n_ghosts:
            if n_trials_this_attempt > max_trials:
                break
            n_trials_this_attempt += 1

            r_pos = np.random.uniform(low=-WIDTH, high=WIDTH,size=(1, 3))
            r_pos = r_pos+COM_BS

            # get distances to other ghost particles
            dists_gho = np.linalg.norm(np.concatenate(rand_positions) - r_pos, axis=1)       

            # check if this ghost particle is close enough to an existing particle
            if dists_gho.min() < gg_max_dist:
                too_close = False
                if len(pdb_pos) > 0:
                    # check if this ghost particle is too close to any system atoms
                    dists_pdb = np.linalg.norm(pdb_pos - r_pos, axis=1)
                    if dists_pdb.min() < min_dist:
                        too_close = True
                        
                # check if this ghost particle is too close to any other ghost atoms
                if dists_gho.min() < gg_min_dist:
                    too_close = True

                if not too_close:
                    rand_positions.append(r_pos)

        if len(rand_positions) == n_ghosts:
            return np.concatenate(rand_positions)

    raise ValueError("Error initializing ghost particles!  Use a smaller number of particles, or a smaller value of min_dist and/or gg_min_dist.")
            
