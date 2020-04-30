import torch
from pharmostat.layers.GSGraph import GSGraph as PGSGraph
from logp.molecular_models.GSGraph import GSGraph as LGSGraph
from pharmostat.layers.utils import adjacency_matrix as padj_dist
from logp.molecular_models.utility import adjacency_matrix as ladj_dist
from pharmostat.layers.utils import distance_matrix as pdist
from logp.molecular_models.utility import distance_matrix as ldist
from pharmostat.layers.utils import fc as pfc
from logp.molecular_models.utility import fc as lfc
from logp.molecular_models.GSGraph import skew as lskew
from logp.molecular_models.GSGraph import kurtosis as lkurtosis
from pharmostat.layers.utils import skew as pskew
from pharmostat.layers.utils import kurtosis as pkurtosis
from logp.molecular_models.GSGraph import moment as lmoment
from pharmostat.layers.utils import moment as pmoment

import numpy as np
import pickle as pkl
# coords = [[0.7852, 0.3557, 0.9492],
#         [0.6310, 0.2493, 0.6343],
#         [0.6927, 0.1372, 0.3046],
#         [0.4866, 0.7781, 0.1464],
#         [0.2730, 0.1045, 0.8313]]


# signals = [[0.3447, 0.9564, 0.8883, 0.7758],
#         [0.5669, 0.1726, 0.1882, 0.9085],
#         [0.2121, 0.7749, 0.6671, 0.6151],
#         [0.9452, 0.6738, 0.7336, 0.5276],
#         [0.6060, 0.4118, 0.7541, 0.3233]]

with open('molecules.pkl', 'rb') as pklf:
    data = pkl.load(pklf)


idx_start = 117
idx_end = 177


#get the coordinates
coords = np.copy(data['coords'][idx_start])

#Define an array of signals, shape: (num_atoms, 3)[cahrge, radius,
#epsilon]

signals = np.copy(data['gaff_signals_notype'][idx_end])


# coords = [[1.0,2.0,3.0], [4.0, 5.0, 6.0], [7.0,8.0,9.0]]
# signals = [[1.0], [2.0], [4.0]]

pgsg = PGSGraph()

lgsg = LGSGraph()

coords = torch.from_numpy(coords)

signals = torch.from_numpy(signals)


# ladjmat = ladj_dist(coords.numpy(), 7.5)
# padjmat = padj_dist(coords, 7.5)

# ppmat = pgsg.lazy_random_walk(padjmat)

# lpmat = lgsg.lazy_random_walk(ladjmat)

# pwavelet = pgsg.graph_wavelet(ppmat)
# lwavelet = lgsg.graph_wavelet(lpmat)

# pzsc = pgsg.zero_order_feature(signals).reshape(-1)
# lzsc = lgsg.zero_order_feature(signals.numpy()).reshape(-1)

# pfsc = pgsg.first_order_feature(pwavelet, signals).reshape(-1)
# lfsc = lgsg.first_order_feature(lwavelet, signals.numpy()).reshape(-1)

# pssc = pgsg.second_order_feature(pwavelet, signals).reshape(-1)
# lssc = lgsg.second_order_feature(lwavelet, signals.numpy()).reshape(-1)


pf = pgsg(coords, signals)
lf = lgsg.molecule_features(coords.numpy(), signals.numpy())

#np.equal(np.around(lfsc, 1), np.around(pfsc.numpy(), 1))

# tlskw = lskew(signals.numpy(), bias=False, axis=0)
# tpskew = pskew(signals, bias=False, axis=0)


# pzsc = pgsg.zero_order_feature(signals).reshape(-1)
# lzsc = lgsg.zero_order_feature(signals.numpy()).reshape(-1)


pf = pgsg(coords, signals).reshape(-1)
lf = lgsg.molecule_features(coords.numpy(), signals.numpy()).reshape(-1)
print(pf)
print(lf)
import ipdb; ipdb.set_trace()
