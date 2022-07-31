
import torch
from torch import nn

class KLDivergenceLoss(nn.Module):
    '''
    This class defines a KL-Divergence based loss, where the difference between the
    predicted distribution and ground truth distribution is calculated and differentiated
    with the goal of minimization.
    The forward function takes in log-probabilities as input and returns mean loss.

    '''
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, predicted_prob, target):
        # takes in log probability as input and caculates loss
        L = -predicted_prob.index_select(-1, target.flatten()).diag()
        # takes mean
        mean_L = L.mean()
        return mean_L

class EstimationError(nn.Module):
    '''
    This class defines the metric used to evaluate the model performance.
    If the predicted age is equal to ground truth, the metric is 0.
    If the predicted age is not equal to ground truth, the metric is (0,1].
    The worst case, the metric is 1.
    '''
    def __init__(self, sigma):
        super(EstimationError, self).__init__()
        self.sigma = sigma

    def forward(self, n, mu):
        std = torch.pow((n - mu), 2) / (2 * self.sigma ** 2)
        std_exp = torch.exp(-std)
        epsilon = 1 - std_exp
        return epsilon


