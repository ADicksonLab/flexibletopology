import torch
import numpy as np
import pickle as pkl
from numpy import array_equal as eq

from flexibletopology.utils.stats import adjacency_matrix as padj_dist
from logp.molecular_models.utility import adjacency_matrix as ladj_dist

from flexibletopology.mlmodels.GSGraph import GSGraph as PGSGraph
from logp.molecular_models.GSGraph import GSGraph as LGSGraph


torch.manual_seed(11)

PRECISION = 3
CUTOFF = 7.5


def test_GSG():

    coords = torch.from_numpy(
        np.array([[0.7852, 0.3557, 0.9492],
                  [0.6310, 0.2493, 0.6343],
                  [0.6927, 0.1372, 0.3046],
                  [0.4866, 0.7781, 0.1464],
                  [0.2730, 0.1045, 0.8313]]))


    signals = torch.from_numpy(
        np.array([[0.3447, 0.9564, 0.8883, 0.7758],
                  [0.5669, 0.1726, 0.1882, 0.9085],
                  [0.2121, 0.7749, 0.6671, 0.6151],
                  [0.9452, 0.6738, 0.7336, 0.5276],
                  [0.6060, 0.4118, 0.7541, 0.3233]]))


    PGSG = PGSGraph()
    LGSG = LGSGraph()

    padjmat = padj_dist(coords, CUTOFF)
    ladjmat = ladj_dist(coords.numpy(), CUTOFF)
    assert eq(np.round(ladjmat, PRECISION),
              np.round(padjmat.numpy(), PRECISION))


    ppmat = PGSG.lazy_random_walk(padjmat)
    lpmat = LGSG.lazy_random_walk(ladjmat)
    assert eq(np.round(ppmat.numpy(), PRECISION),
              np.round(lpmat, PRECISION))

    pwavelet = PGSG.graph_wavelet(ppmat)
    lwavelet = LGSG.graph_wavelet(lpmat)
    assert eq(np.round(pwavelet.numpy(), PRECISION),
              np.round(lwavelet, PRECISION))


    pzsc = PGSG.zero_order_feature(signals).reshape(-1)
    lzsc = LGSG.zero_order_feature(signals.numpy()).reshape(-1)
    assert eq(np.round(pzsc.numpy(), PRECISION),
              np.round(lzsc, PRECISION))

    pfsc = PGSG.first_order_feature(pwavelet, signals).reshape(-1)
    lfsc = LGSG.first_order_feature(lwavelet, signals.numpy()).reshape(-1)
    assert eq(np.round(pfsc.numpy(), PRECISION),
              np.round(lfsc, PRECISION))

    pssc = PGSG.second_order_feature(pwavelet, signals).reshape(-1)
    lssc = LGSG.second_order_feature(lwavelet, signals.numpy()).reshape(-1)
    assert eq(np.round(pssc.numpy(), PRECISION),
              np.round(lssc, PRECISION))


    pfeatures = PGSG(coords, signals).reshape(-1)
    lfeatures = LGSG.molecule_features(coords.numpy(), signals.numpy()).reshape(-1)
    assert eq(np.round(pfeatures.numpy(), PRECISION),
              np.round(lfeatures, PRECISION))
