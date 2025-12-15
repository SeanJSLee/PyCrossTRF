import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Literal, Tuple, Any


def kde_torch_gaussian(ser:pd.Series, 
                       device: Literal['cpu','cuda'] = 'cuda',
                       verbose:bool = False,
                       bandwidths = np.logspace(-2.5, -1, 40)
                       )-> (np.array, np.array, float) :
    # Check if CUDA is available and set the device
    if device == 'cuda' :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 
    data = np.array(ser).reshape(-1,1)
    N = data.__len__()
    data_torch = torch.from_numpy(data).float().to(device)

    # bandwidth candidate - choose smaller than predicted scale '.001' to its double.
    # bandwidths = np.logspace(np.log(.0005), np.log(.002), 50)
    # bandwidths = np.logspace(-2.5, -1, 40)

    # search for bandwidth
    scores = []
    for h in bandwidths:
        score = loocv_score_pytorch(data_torch, h)
        scores.append(score.item())
    # 
    best_bandwidth = bandwidths[np.argmax(scores)]
    if verbose : 
        print(f'Using device: {device}')
        print(f'{best_bandwidth:.5f}')

    # generate estimation 
    quantiles = np.linspace(0,1,1001)
    quantiles_torch = torch.from_numpy(quantiles).float().to(device)
    density_torch = kde_pytorch(data_torch, quantiles_torch, best_bandwidth)
    density = density_torch.cpu().numpy()
    # 
    return quantiles, density, best_bandwidth



def gaussian_kernel(x, bandwidth):
    'Gaussian kernel for KDE, with a constant for the normal distribution.'
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * torch.exp(-0.5 * (x / bandwidth)**2)

def kde_pytorch(x_data, x_eval, bandwidth):
    'Computes Gaussian KDE using PyTorch for GPU acceleration.'
    diff = x_data.view(-1, 1) - x_eval.view(1, -1)
    kernel_vals = gaussian_kernel(diff, bandwidth)
    return torch.mean(kernel_vals, dim=0)

def loocv_score_pytorch(x_data, bandwidth):
    'Computes the log-likelihood score using Leave-One-Out Cross-Validation.'
    n = x_data.shape[0]
    diff = x_data.view(-1, 1) - x_data
    kernel_vals = gaussian_kernel(diff, bandwidth)
    kernel_vals.fill_diagonal_(0)
    f_minus_i = torch.sum(kernel_vals, dim=1) / (n - 1)
    return torch.sum(torch.log(f_minus_i + 1e-10))



