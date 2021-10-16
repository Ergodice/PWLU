'''
Daniel Monroe, 2021
Implementation of paper
"Piecewise Linear Unit (PWLU) Activation Function"
 https://arxiv.org/pdf/2104.03693.pdf

Trainable boundaries are foregone in place of batch norms.
To simplify the implementation, there are fixed boundaries which are the same for all channels.
Instead of training slopes, we consider the regions outside of the boundaries to be extensions of the outermost regions.
'''

from utils import *
import torch
from torch.nn import Parameter
import torch.nn as nn
import abc
import numpy as np

__all__ = ['PWLU', 'PWLUBase', 'RegularizedPWLU', 'normed_pwlu']


def normalize(points: torch.Tensor, fix_std: bool = True) -> None:
    '''
    Normalize points to have mean 0 and std 1
    :param points: points to normalize
    :param fix_std: if true, will normalize to have std 1, default true
    '''
    with torch.no_grad():
        if len(points.shape) == 2:
            if fix_std:
                nn.functional.normalize(points, out=points)
            points -= torch.mean(points, dim=1, keepdim=True)
        else:

            if fix_std:
                nn.functional.normalize(points, dim=0, out=points)
            points -= torch.mean(points)


# @torch.jit.script


def pwlu_forward(x: torch.Tensor, points: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    '''
    Apply PWLU activation function to x with points and bound
    :param x: input
    :param points: points to evaluate PWLU at
    :param bounds: bounds of PWLU
    '''

    channelwise = len(points.shape) == 2
    n_regions = points.shape[1 if channelwise else 0] - 1
    batch_size, n_channels, *other_dims = x.shape

    # values for bounds will be mapped to 0 and 1, some will lie outside
    if channelwise:
        if len(bounds.shape) == 1:
            x_normal = ((x - bounds[0]) / (bounds[1] - bounds[0])).moveaxis(1, 0)
        else:
            x_channels_last = x.moveaxis(1, -1)
            x_normal = ((x_channels_last - bounds[..., 0]) / (bounds[..., 1] - bounds[..., 0])).moveaxis(-1, 0)
    else:
        x_normal = (x - bounds[0]) / (bounds[1] - bounds[0])

    # regions from 0 to self.n_regions - 1 that values belong in
    regions = (x_normal.clamp(0, .999) * n_regions).floor()

    # create tensor of PWLU regions into shape (channels, ...) if channelwise else (...)
    dists = x_normal * n_regions - regions

    regions_packed = regions.long()
    if channelwise:
        regions_packed = regions_packed.reshape(n_channels, -1)
    else:
        regions_packed = regions_packed.reshape(-1)

    # create tensors of left and right points of regions
    left_points = torch.gather(points, -1, regions_packed)
    right_points = torch.gather(points, -1, regions_packed + 1)
    if channelwise:
        left_points = left_points.reshape(n_channels, batch_size, *other_dims)
        right_points = right_points.reshape(n_channels, batch_size, *other_dims)

        left_points = left_points.moveaxis(0, 1)
        right_points = right_points.moveaxis(0, 1)
        dists = dists.moveaxis(0, 1)

    else:
        left_points = left_points.reshape(x.size())
        right_points = right_points.reshape(x.size())

    # calculate activation
    ret = left_points * (1 - dists) + right_points * dists

    '''
    # Second order pwlu
    if qs:  
            assert False, "second order  not implemented"
            q_strengths = torch.gather(4 * self.qs / bound, -1, regions_packed)

            if channelwise:
                q_strengths = q_strengths.reshape(n_channels, h, w, batch_size)
                q_strengths = q_strengths.moveaxis(-1, 0)
            else:
                q_strengths = q_strengths.reshape(batch_dims)
            ret += q_strengths * dists * (dists - 1)
    '''

    return ret


class PWLUBase(torch.nn.Module, abc.ABC):
    '''
    Abstract base class for PWLU
    Accepts n_regions, bound, init, norm, and norm_args arguments to be passed in from child class
    '''

    def __init__(self,
                 n_regions: int = 12,
                 bound: float = 2.5,
                 learnable_bound: bool = False,
                 same_bound: bool = False,
                 init='relu',
                 normed: bool = False,
                 norm_args: dict = {},
                 **kwargs):
        '''
        :param n_regions: number of regions in the PWLU
        :param n_channels: number of channels in input
        :param bound: bound of the PWLU
        :param learnable_bound: if true, bound is learnable
        :param same_bound: if true, all channels have the same bound
        :param init: function to initialize the points
        :param norm: normalization layer, defaults to BatchNorm2d
        :param norm_args: arguments to pass to norm
        :param kwargs: other arguments, will raise exception if not empty
        '''

        super().__init__()

        if kwargs:
            raise ValueError(f'Unused keyword arguments: {kwargs}')

        assert hasattr(self, '_n_channels'), 'PWLUBase subclass must set _n_channels before calling super().__init__'

        if self.channelwise and not same_bound:
            bound = torch.Tensor([[-bound, bound]])
            bound = bound.repeat(self._n_channels, 1)
        else:
            bound = torch.Tensor([-bound, bound])

        self._bounds = Parameter(bound)
        self._bounds.requires_grad = learnable_bound
        self._n_regions = n_regions
        self._n_points = n_regions + 1
        self.set_points(get_activation(init))

        # Create normalization layer
        self._norm = None
        if normed:
            self._norm = nn.LazyBatchNorm2d(affine=False, **norm_args)

    def forward(self, x: torch.Tensor, use_norm=True):
        if use_norm and self._norm(x) is not None:
            x = self._norm(x)
        return pwlu_forward(x, self.get_points(), self._bounds)

    def __repr__(self):
        ret = f'{self.__class__}(n_regions={self._n_regions}, bound={self._bounds})'
        if self.channelwise:
            ret += f'n_channels={self._n_channels}'
        return ret

    def get_plottable(self, n_points: int = 100, bound: float = 3) -> torch.Tensor:
        '''
        :param n_points: number of points to plot
        :param bound: bound to plot, defaults to 3
        '''
        self.normalize_points()

        locs = np.linspace(-bound, bound, n_points)[np.newaxis, np.newaxis, :]
        if self.channelwise:
            locs = locs.repeat(self._n_channels, 1)
        with torch.no_grad():
            try:
                vals = self.forward(torch.from_numpy(locs).cuda(), use_norm=False)
            except:
                vals = self.forward(torch.from_numpy(locs), use_norm=False)
        vals = vals.cpu().numpy()
        return locs.squeeze(), vals.squeeze()

    def n_channels(self) -> int:
        return self._n_channels

    def normalize_points(self) -> None:
        pass

    @abc.abstractmethod
    def get_points(self):
        pass

    @abc.abstractmethod
    def set_points(self):
        pass

    @property
    def channelwise(self) -> bool:
        return self._n_channels is not None

    @property
    def n_regions(self) -> int:
        return self._n_regions

    @property
    def bound(self) -> float:
        return self._bound

    @property
    def points(self) -> torch.Tensor:
        return self.get_points()

    @property
    def norm(self) -> torch.nn.Module:
        return self._norm

    @property
    def init(self):
        return self._init


class PWLU(PWLUBase):
    '''
    Vanilla implementation of the PWLU.
    The PWLU is initialized as channelwise if n_channels is specified; otherwise, it is layerwise.
    Other parameters are pased to PWLUBase.
    '''

    def __init__(self,
                 n_channels: int = None,
                 **kwargs):
        '''
        :param n_channels: number of channels in input, defaults to None
        :param kwards: see PWLUBase
        '''
        self._n_channels = n_channels
        super().__init__(**kwargs)
        self.set_points(self._init)

    def forward(self, x: torch.Tensor):
        return pwlu_forward(x, self._points, self._bounds)

    def normalize_points(self) -> None:
        normalize(self._points)

    def set_points(self, init) -> None:
        self._init = init
        bound = torch.flatten(self._bounds)[0].item()
        locs = torch.linspace(-bound, bound, self._n_points)
        if self.channelwise:
            locs = locs.repeat(self._n_channels, 1)
        self._points = Parameter(init(locs))
        self.normalize_points()

    def get_points(self) -> torch.Tensor:
        return self._points


class RegularizedPWLU(PWLUBase):
    '''
    The regularized PWLU is a a stable version of the channelwise PWLU meant for smaller networks.
    It is channelwise by definition and has a "master points" parameter which is the average of the other channel units
    and "deviation points" parameters which are the deviations from the master points, controlled in their strength by the relative_factor parameter.
    Other parameters are pased to PWLUBase.
    '''

    def __init__(self,
                 n_channels: int,
                 *,
                 relative_factor=0.5,
                 **kwargs):
        '''
        :param n_channels: number of channels in input
        :param relative_factor: how much to scale the relative point deviations by
        :param kwargs: see PWLUBase
        '''

        self._n_channels = n_channels
        super().__init__(same_bound=True, **kwargs)
        self._relative_factor = relative_factor
        self._master_points = Parameter(torch.zeros(self._n_points))
        self._relative_points = Parameter(torch.zeros(self._n_channels, self._n_points))
        self.set_points(self._init)

    def normalize_points(self) -> None:
        normalize(self._master_points)
        normalize(self._relative_points, fix_std=False)

    def get_points(self) -> torch.Tensor:
        return self._master_points + self._relative_points * self._relative_factor

    def set_points(self, init) -> None:
        self._init = init

        bound = torch.flatten(self._bounds)[0].item()
        master_locs, locs = (torch.linspace(-bound, bound, self._n_points) for _ in range(2))
        locs = locs.repeat(self._n_channels, 1)
        with torch.no_grad():
            self._master_points = Parameter(init(master_locs))
            self._relative_points = Parameter(torch.zeros_like(locs))
        self.normalize_points()


def normed_pwlu(pwlu_class: type, *args, norm: nn.Module, norm_args: dict = {}, **kwargs) -> PWLUBase:
    '''
    Make a PWLU with a normalization layer.
    :param pwlu_class: PWLU class to instantiate
    :param args: positional arguments to pass to PWLU class, usually just num channels
    :param norm: normalization layer
    :param norm_args: arguments to pass to normalization layer
    :param kwargs: keyword arguments to pass to PWLU class
    '''
    return pwlu_class(*args, norm=norm, norm_args=norm_args, **kwargs)


if __name__ == '__main__':
    args = [
        {'n_channels': 10, 'learnable_bound': True, 'same_bound': True},
        {'n_channels': 10, 'learnable_bound': False, 'same_bound': False},
        {'learnable_bound': False, 'same_bound': False},
        {},
    ]
    for kwargs in args:
        pwlu = PWLU(**kwargs)
        assert pwlu(torch.zeros(3, 10, 17, 19)).shape == torch.Size([3, 10, 17, 19])

    pwlu = RegularizedPWLU(10, learnable_bound=True)
    assert pwlu(torch.zeros(3, 10, 17, 19)).shape == torch.Size([3, 10, 17, 19])
