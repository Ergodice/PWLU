'''
Daniel Monroe, 2021
Useful functions and classes
'''

import torch.nn as nn
import torch

__all__ = ['get_activation', 'SquareActivation', 'IdentityActivation']


class SquareActivation(nn.Module):
    '''
    Square activation function, clamps the output between 0 and 20 to avoid overflow
    '''
    def forward(self, x):
        return torch.clamp(x ** 2, 0, 20)

class IdentityActivation(nn.Module):
    '''
    Identity activation function
    '''
    def forward(self, x):
        return x

def get_activation(activation):
    '''
    Get activation function with name activation or return activation if it is already a function
    :param activation: activation function name or activation function
    '''
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
        except:
            activation = activation()
        return activation

