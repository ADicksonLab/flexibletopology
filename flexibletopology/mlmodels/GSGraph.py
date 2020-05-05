import torch
from torch import nn
import numpy as np

from flexibletopology.utils.utils import adjacency_matrix, skew, kurtosis


class GSGraph(nn.Module):

    def __init__(self, wavelet_num_steps=4, radial_cutoff=7.5,
                 features=(True, True, True)):
        super(GSGraph, self).__init__()
        #self.is_trainable = False
        self.wavelet_num_steps = wavelet_num_steps
        self.radial_cutoff = radial_cutoff
        self.features = features

    def lazy_random_walk(self, adj_mat):

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
    def graph_wavelet(self, probability_mat):

        # 2^j
        steps = []
        for step in range(self.wavelet_num_steps):

            steps.append(2 ** step)

        wavelets = []
        for i, j in enumerate(steps):
            wavelet = torch.matrix_power(probability_mat, j) \
                - torch.matrix_power(probability_mat, 2*j)

            wavelets.append(wavelet)

        return torch.stack(wavelets)

    def zero_order_feature(self, signals):
        #zero order feature calcuated using signal of the graph.
        features = []

        features.append(torch.mean(signals, axis=0))
        features.append(torch.var(signals, axis=0, unbiased=False))
        features.append(skew(signals, axis=0, bias=False))
        features.append(kurtosis(signals, axis=0, bias=False))

        return torch.stack(features).reshape(-1, 1)

    def first_order_feature(self, wavelets, signals):

        wavelet_signals = torch.abs(torch.matmul(wavelets, signals))
        features = []
        features.append(torch.mean(wavelet_signals, axis=1))
        features.append(torch.var(wavelet_signals, axis=1, unbiased=False))
        features.append(skew(wavelet_signals, axis=1, bias=False))
        features.append(kurtosis(wavelet_signals, axis=1, bias=False))

        return torch.stack(features).reshape(-1, 1)

    def second_order_feature(self, wavelets, signals):
        wavelet_signals = torch.abs(torch.matmul(wavelets, signals))
        coefficents = []
        for i in range(1, len(wavelets)):
            coefficents.append(torch.einsum('ij,ajt ->ait', wavelets[i],
                                            wavelet_signals[0:i]))


        coefficents = torch.abs(torch.cat(coefficents, axis=0))

        features = []

        features.append(torch.mean(coefficents, axis=1))
        features.append(torch.var(coefficents, axis=1, unbiased=False))
        features.append(skew(coefficents, axis=1, bias=False))
        features.append(kurtosis(coefficents, axis=1, bias=False))

        return torch.stack(features).reshape(-1, 1)

    def wavelets(self, adj_mat):

        probability_mat = self.lazy_random_walk(adj_mat)

        return self.graph_wavelet(probability_mat)


    def forward(self, positions, signals):

        adj_mat = adjacency_matrix(positions, self.radial_cutoff)

        probability_mat = self.lazy_random_walk(adj_mat)

        wavelets = self.graph_wavelet(probability_mat)

        gsg_features = []

        if self.features[0]:
            gsg_features.append(self.zero_order_feature(signals))

        if self.features[1]:
            gsg_features.append(self.first_order_feature(wavelets, signals))

        if self.features[2]:
            gsg_features.append(self.second_order_feature(wavelets, signals))


        return  torch.cat(gsg_features, axis=0)
