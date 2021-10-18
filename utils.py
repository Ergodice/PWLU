"""
Daniel Monroe, 2021
Useful functions and classes
"""

import torch.nn as nn
import torch


def normalize(points: torch.Tensor, fix_std: bool = True) -> None:
    """
    Normalize points to have mean 0 and std 1
    :param points: points to normalize
    :param fix_std: if true, will normalize to have std 1, default true
    """
    with torch.no_grad():
        if len(points.shape) == 2:
            if fix_std:
                nn.functional.normalize(points, out=points)
            points -= torch.mean(points, dim=1, keepdim=True)
        else:
            if fix_std:
                nn.functional.normalize(points, dim=0, out=points)
            points -= torch.mean(points)


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
