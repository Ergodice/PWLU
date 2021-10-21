"""
Daniel Monroe, 2021
Useful functions and classes
"""

import torch.nn as nn
import torch
import functools


@functools.lru_cache(128)
def make_inverse_matrix(n: int, device = 'cpu'):
    """
    Makes matrix used to optimize quadratic approximator
    :param n: number of points
    :param device: device to use
    """

    # Calculate sums of nth powers from 0 to n-1
    s4 = n * (n - 1) * (2 * n - 1) * (3 * n**2 - 3*n - 1) / 30
    s3 = (n-1) ** 2 * n**2 / 4
    s2 = n * (n - 1) * (2 * n - 1) / 6
    s1 = n * (n - 1) / 2

    # Make matrix
    matrix_elements = [[s4, s3, s2],
                       [s3, s2, s1],
                       [s2, s1, n]]
    matrix = torch.tensor(matrix_elements, dtype=torch.float, device=device)
    return torch.linalg.inv(matrix)

def calculate_abcs(points: torch.Tensor):
    """
    Calculate a, b, c for quadratic approximation
    Acceptable explanation given here: https://www.tutorialspoint.com/statistics/quadratic_regression_equation.htm
    :param points: points to calculate on
    """
    with torch.no_grad():
        if len(points.shape) == 1:
            points = points.unsqueeze(0)
        n = points.shape[-1]
        arange = torch.arange(n, device=points.device, dtype=torch.float)
        arange_squared = arange ** 2
        top = torch.einsum('ij, j -> i', points, arange_squared)
        middle = torch.einsum('ij, j -> i', points, arange)
        bottom = torch.einsum('ij -> i', points)
        vectors = torch.stack([top, middle, bottom], dim=1)
        inverse_matrix = make_inverse_matrix(n, points.device)
        ret = inverse_matrix @ vectors.transpose(0, 1)
    return ret

def approximate_quadratic(points: torch.Tensor):
    """
    Approximate points with a quadratic function
    :param points: points to approximate
    """
    if len(points.shape) == 1:
        points = points.unsqueeze(0)
    n_points = points.shape[-1]
    a, b, c = calculate_abcs(points)
    arange = torch.arange(n_points)
    return a * arange ** 2 + b * arange + c

def approximate_quadratic_piecewise(points: torch.Tensor, n: int):
    """
    Replace every n points with a quadratic approximation of those points
    :param points points to approximate
    :param n number of pointss to take at a time
    """

    # TODO: Allow n to be any number >=3 (don't need n_points % n >=3 )
    assert n >= 3, 'n must be at least 3 to approximate'

    if len(points.shape) == 1:
        points = points.unsqueeze(0)
    n_points = points.shape[-1]
    start = 0

    while start < n_points:
        with torch.no_grad():
            end = min(start + n, n_points)
            points[..., start:end] = approximate_quadratic(points[..., start:end])
            start = end


def normalize(points: torch.Tensor, fix_std: bool = True) -> float:
    """
    Normalize points to have mean 0 and std 1
    :param points: points to normalize
    :param fix_std: if true, will normalize to have std 1, default true
    """

    with torch.no_grad():
        start_var = torch.var(points).item()
        if len(points.shape) == 2:
            points -= torch.mean(points, dim=1, keepdim=True)
            if fix_std:
                nn.functional.normalize(points, out=points)
        else:
            points -= torch.mean(points)
            if fix_std:
                nn.functional.normalize(points, dim=0, out=points)


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
    values = torch.arange(5, dtype=torch.float) + torch.arange(5, dtype=torch.float) ** 2
    approximate_quadratic_piecewise(values, 5)
    print(values)