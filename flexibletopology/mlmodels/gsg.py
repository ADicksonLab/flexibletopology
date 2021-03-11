import torch
from torch import nn
import numpy as np
from torch import Tensor
from typing import Tuple, Optional, NamedTuple

from flexibletopology.utils.stats import adjacency_matrix, skew, kurtosis

class GSG(nn.Module):

    def __init__(self, max_wavelet_scale: int=4, radial_cutoff: float=0.52,
                 sm_operators: Tuple[bool, bool, bool]=(True, True, True)):

        super().__init__()
        self.is_trainable = False
        self.max_wavelet_scale = max_wavelet_scale
        self.radial_cutoff = radial_cutoff
        self.sm_operators = sm_operators

    def lazy_random_walk(self, adj_mat: Tensor) -> Tensor:

        # calcuate degree matrix
        degree_mat = torch.sum(adj_mat, dim=0)

        # calcuate A/D
        adj_degree = torch.div(adj_mat, degree_mat)

        # sets NAN vlaues to zero
        #adj_degree = np.nan_to_num(adj_degree)

        identity =  torch.zeros_like(adj_degree)
        identity.fill_diagonal_(1.0)

        return 1/2 * (identity + adj_degree)

    #calcuate the graph wavelets based on the paper
    def graph_wavelet(self, probability_mat:Tensor) -> Tensor:

        wavelets = []
        for j in range(self.max_wavelet_scale):
            wavelet = torch.matrix_power(probability_mat, int(2 ** j)) \
                - torch.matrix_power(probability_mat, int(2*(2 ** j)))

            wavelets.append(wavelet)

        return torch.stack(wavelets)

    def zero_order_feature(self, signals) -> Tensor:
        #zero order feature calcuated using signal of the graph.
        features = []

        features.append(torch.mean(signals, dim=0))
        features.append(torch.var(signals, dim=0, unbiased=False))
        features.append(skew(signals, dim=0, bias=False))
        features.append(kurtosis(signals, dim=0, bias=False))

        return torch.stack(features).reshape(-1, 1)

    def first_order_feature(self, wavelets:Tensor, signals:Tensor) -> Tensor:

        wavelet_signals = torch.abs(torch.matmul(wavelets, signals))
        features = []
        features.append(torch.mean(wavelet_signals, dim=1))
        features.append(torch.var(wavelet_signals, dim=1, unbiased=False))
        features.append(skew(wavelet_signals, dim=1, bias=False))
        features.append(kurtosis(wavelet_signals, dim=1, bias=False))

        return torch.stack(features).reshape(-1, 1)

    def second_order_feature(self, wavelets: Tensor, signals: Tensor) -> Tensor:
        wavelet_signals = torch.abs(torch.matmul(wavelets, signals))
        coefficents = []
        for i in range(1, len(wavelets)):
            coefficents.append(torch.einsum('ij,ajt ->ait', wavelets[i],
                                            wavelet_signals[0:i]))


        coefficents = torch.abs(torch.cat(coefficents, dim=0))

        features = []

        features.append(torch.mean(coefficents, dim=1))
        features.append(torch.var(coefficents, dim=1, unbiased=False))
        features.append(skew(coefficents, dim=1, bias=False))
        features.append(kurtosis(coefficents, dim=1, bias=False))

        return torch.stack(features).reshape(-1, 1)

    def wavelets(self, adj_mat: Tensor) -> Tensor:

        probability_mat = self.lazy_random_walk(adj_mat)

        return self.graph_wavelet(probability_mat)


    def forward(self, positions: Tensor, signals: Tensor) -> Tensor:

        adj_mat = adjacency_matrix(positions, self.radial_cutoff)

        probability_mat = self.lazy_random_walk(adj_mat)

        wavelets = self.graph_wavelet(probability_mat)

        gsg_features = []


        if self.sm_operators[0]:
            gsg_features.append(self.zero_order_feature(signals))

        if self.sm_operators[1]:
            gsg_features.append(self.first_order_feature(wavelets, signals))

        if self.sm_operators[2]:
            gsg_features.append(self.second_order_feature(wavelets, signals))


        return torch.cat(gsg_features, dim=0)
