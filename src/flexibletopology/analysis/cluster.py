import numpy as np
from sklearn.cluster import KMeans
from flexibletopology.analysis.grid import Grid

def cluster_elec_pot(states,grid,verbose=False,n_clusters=10):
    if verbose:
        print("Calculating electrostatic potentials..")
    to_cluster_list = []
    for s in states:
        to_cluster_list.append(grid.compute_elec_pot_grid(s).flatten())

    if verbose:
        print(f"Fitting to {n_clusters} clusters with KMeans..")
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(to_cluster_list)

    if verbose:
        print(f"Finding most populated cluster..")

    counts,_ = np.histogram(labels,bins=n_clusters,range=(0,n_clusters))

    close_dists=[None for i in range(n_clusters)]
    close_idxs=[None for i in range(n_clusters)]
    for i,state_vec in enumerate(to_cluster_list):
        dist = np.sum(np.square(state_vec - kmeans.cluster_centers_[labels[i]]))
        if close_dists[labels[i]] is None:
            close_dists[labels[i]] = dist
            close_idxs[labels[i]] = i
        else:
            if close_dists[labels[i]] > dist:
                close_dists[labels[i]] = dist
                close_idxs[labels[i]] = i
                
    return counts, kmeans.cluster_centers_, close_idxs
    
    
