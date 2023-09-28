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


def gen_init_pos(n_ghosts, COM_BS, WIDTH, pdb_pos, min_dist=0.1, gg_min_dist=0.1):

    rand_positions = []
    n_attempts = 0
    
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

        if np.all(dists_pdb > min_dist) and np.all(dists_gho > gg_min_dist):
            rand_positions.append(r_pos)

    return np.concatenate(rand_positions)
