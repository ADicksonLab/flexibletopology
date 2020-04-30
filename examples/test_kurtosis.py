from pharmostat.layers.utils import moment
import  scipy.stats as stats
from pharmostat.layers.utils import kurtosis
import torch
a = torch.rand(6, 4, 3)*100
print(moment(a, 4,1))
print(stats.moment(a.numpy(), 4, 1))
print(kurtosis(a, axis=1, bias=False))
print(stats.kurtosis(a.numpy(), axis=1, bias=False))
