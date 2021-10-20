"""
Daniel Monroe, 2021
Useful functions and classes
"""

import torch.nn as nn
import torch
from functools import lru_cache
from math import sqrt


@lru_cache(100)
def _make_smooth_matrix(size: int, factor: float, device='cpu') -> torch.Tensor:
    """
    Cached: make a matrix size x with entries factor**abs(row-column), with rows normalized to have sum 1
    :param size: size of matrix
    :param factor: factor to multiply by
    :param device: device to put matrix on
    """
    size_range = torch.arange(size, device=device)
    matrix = factor ** torch.abs(size_range.view(1, -1) - size_range.view(-1, 1))
    # Divide each row of matrix by its sum
    matrix = matrix / torch.sum(matrix, dim=1, keepdim=True)
    return matrix


def make_smooth_matrix(size: int, factor: float, device='cpu') -> torch.Tensor:
    """
    Make a matrix size x with entries factor**abs(row-column), with rows normalized to have sum 1
    :param size: size of matrix
    :param factor: factor to multiply by
    :param device: device to put matrix on
    """
    return _make_smooth_matrix(size, round(factor, 2), device)


def normalize(points: torch.Tensor, fix_std: bool = True) -> float:
    """
    Normalize points to have mean 0 and std 1
    :param points: points to normalize
    :param fix_std: if true, will normalize to have std 1, default true
    """

    with torch.no_grad():
        start_var = torch.var(points).item()
        if len(points.shape) == 2:
            if fix_std:
                nn.functional.normalize(points, out=points)
            points -= torch.mean(points, dim=1, keepdim=True)
        else:
            if fix_std:
                nn.functional.normalize(points, dim=0, out=points)
            points -= torch.mean(points)
        end_var = torch.var(points).item()
    if start_var < 10 ** -4:
        return 1
    return sqrt(start_var / end_var)


class SquareActivation(nn.Module):
    """
    Square activation function, clamps the output between 0 and 20 to avoid overflow
    """

    @staticmethod
    def forward(x):
        return torch.clamp(x ** 2, 0, 20)


class IdentityActivation(nn.Module):
    """
    Identity activation function
    """

    @staticmethod
    def forward(x):
        return x


def get_activation(activation):
    """
    Get activation function with name activation or return activation if it is already a function
    :param activation: activation function name or activation function
    """
    activations = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'gelu': nn.GELU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        None: None,
        'none': None,
        'linear': IdentityActivation,
        'identity': IdentityActivation,
        'square': SquareActivation,
    }
    assert activation != 'pwlu', 'pwlu is deprecated, use PWLU instead'

    if isinstance(activation, type):
        return activation()
    elif callable(activation):
        return activation
    else:
        if activation not in activations:
            raise ValueError(f'Activation {activation} not supported')
        activation = activations[activation]
        if activation is None:
            return None
        try:
            activation = activation(inplace=True)
        except TypeError:
            activation = activation()

        return activation


if __name__ == '__main__':
    print(make_smooth_matrix(5, .5))
