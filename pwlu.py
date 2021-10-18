'''
Daniel Monroe, 2021
Implementation of paper
"Learning specialized activation functions with the Piecewise Linear Unit"
https://arxiv.org/pdf/2104.03693.pdf
The structure is exactly the same as the original, but the implementation is marginally faster.
'''

from utils import *
import torch
from torch.nn import Parameter
import torch.nn as nn
from abc import abstractmethod, ABC
import numpy as np
import time
import logging

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

def pwlu_forward(x: torch.Tensor, points: torch.Tensor, bounds: torch.Tensor, left_slopes: torch.Tensor,
                 right_slopes: torch.Tensor, compiled_slopes: torch.Tensor = None,
                 points_is_compiled: bool = False) -> torch.Tensor:
    '''
    Apply PWLU activation function to x with points and bound
    :param x: input
    :param points: points to evaluate PWLU at
    :param bounds: bounds of PWLU
    :param left_slopes: left slopes of PWLU
    :param right_slopes: right slopes of PWLU
    :param slopes_compiled: compiled slopes of PWLU if already computed
    :param points_is_compiled: if true, points is pre-compiled, i.e., as false_points
    '''

    # Initialize variables
    batch_size, n_channels, *other_dims = x.shape
    channelwise = len(points.shape) == 2
    n_points = points.shape[-1]
    if points_is_compiled:
        # Simulated points are pre-compiled
        n_points -= 1
    
    n_regions = n_points - 1
    left_bounds = bounds[..., 0]
    right_bounds = bounds[..., 1]
    region_lengths = (right_bounds - left_bounds) / n_regions

    # Compute slopes
    if compiled_slopes is not None:
        slopes = compiled_slopes
    else:
        slopes = (points - torch.roll(points, 1, dims=-1))[..., 1:] / region_lengths.unsqueeze(-1)
        slopes = torch.cat([left_slopes.unsqueeze(-1), slopes, right_slopes.unsqueeze(-1)], dim=-1)


    # Create normalized version of x; values for bounds will be mapped to 0 (sim_left_bound) and 1 (right bound), some will lie outside
    sim_left_bounds = left_bounds - region_lengths
    if channelwise:
        if len(region_lengths.shape) == 0:
            x_normal = (x - sim_left_bounds) / ((n_regions + 1) * region_lengths)
            x_normal = x_normal.moveaxis(1, 0)

        else:
            x_channels_last = x.moveaxis(1, -1)
            x_normal = (x_channels_last - sim_left_bounds) / ((n_regions + 1) * region_lengths)
            x_normal = x_normal.moveaxis(-1, 0)

    else:
        x_normal = (x - sim_left_bounds) / ((n_regions + 1) * region_lengths)

    # Regions are 0, 1... n_regions, n_regions + 1; outermost are out of bounds
    regions = (x_normal.clamp(0, 1.001) * (n_regions + 1)).floor()

    # Create tensor of dists from 0 to 1 from left point: shape (channels, ...) if channelwise else (...)
    dists = (x_normal * (n_regions + 1) - regions)
    if len(region_lengths.shape) == 0:
        dists *= region_lengths
    else:
        dists = dists.moveaxis(0, -1)
        dists *= region_lengths
        dists = dists.moveaxis(-1, 0)

    # Pack regions into form suitable for evaluation
    regions_packed = regions.long()
    if channelwise:
        regions_packed = regions_packed.reshape(n_channels, -1)
    else:
        regions_packed = regions_packed.reshape(-1)

    # Create tensors of points including simulated left point
    if points_is_compiled:
        false_points = points
    else:
        false_points = torch.cat([(points[..., 0] - left_slopes * region_lengths).unsqueeze(-1), points], dim=-1)

    # Compute left_points and slope necessary for evaluation
    left_points = torch.gather(false_points, -1, regions_packed)
    slopes = torch.gather(slopes, -1, regions_packed)

    # Compute activations
    if channelwise:
        left_points = left_points.reshape(n_channels, batch_size, *other_dims)
        slopes = slopes.reshape(n_channels, batch_size, *other_dims)

        ret = left_points + dists * slopes
        ret = ret.moveaxis(0, 1)

    else:
        left_points = left_points.reshape(x.size())
        slopes = slopes.reshape(x.size())
        ret = left_points + dists * slopes

    return ret


class PWLUBase(torch.nn.Module, ABC):
    '''
    Abstract base class for PWLU
    The number of regions represents the number of regions inside the bounds.
    There are always parameters for controlling slopes outside bounds.
    Accepts n_regions, bound, learnable_bound, same_bound, init, norm, and norm_args arguments to be passed in from child class
    '''

    def __init__(self,
                 n_regions: int = 6,
                 bound: float = 2.5,
                 learnable_bound: bool = False,
                 same_bound: bool = False,
                 init='relu',
                 normed: bool = False,
                 norm_args: dict = {},
                 autocompile=True,
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
        :param autocompile: if true, compile the forward function
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

        # Compile settings
        self._autocompile = autocompile
        self._compiled_points = None
        self._compiled_slopes = None
        self._compiled = False

        self.set_points(get_activation(init))

        # Create normalization layer
        self._norm = nn.LazyBatchNorm2d(affine=False, **norm_args) if normed else None
    

    def to(self, *args, **kwargs):
        self._compiled = False
        return super().to(*args, **kwargs)

    def cuda(self):
        self._compiled = False
        return super().to()

    def cpu(self):
        self._compiled = False
        return super().cpu()

    def compile_for_eval(self):
        '''
        Compile the points and slopes for evaluation
        The compiled points includes the effective point left of the left bound
        '''
        left_bounds = self._bounds[..., 0]
        right_bounds = self._bounds[..., 1]
        region_lengths = (right_bounds - left_bounds) / self._n_regions
        points = self.get_points()
        self._compiled_points = torch.cat([(points[..., 0] - self._left_slopes * region_lengths).unsqueeze(-1), points],
                                          dim=-1).detach().to(points.device)

        slopes = (points - torch.roll(points, 1, dims=-1))[..., 1:] / region_lengths.unsqueeze(-1)
        slopes = torch.cat([self._left_slopes.unsqueeze(-1), slopes, self._right_slopes.unsqueeze(-1)], dim=-1)
        self._compiled_slopes = slopes.detach().to(points.device)

        self._compiled = True

    def forward(self, x: torch.Tensor, normed: bool = True):
        '''
        Forward function of the BasePWLU
        :param x: input tensor
        :param normed: if true, use self._norm on the input
        '''
        norm = self._norm if normed else None
        if self.training and not self._autocompile:
            # Compile as if in training mode
            self._compiled = False
            return pwlu_forward(norm(x) if norm else x, self.get_points(), self._bounds, self._left_slopes,
                                self._right_slopes)
        else:
            # Compile points and slopes if not compiled, run using compiled form
            if not self._compiled:
                self.compile_for_eval()
            try:
                return pwlu_forward(norm(x) if norm else x, self._compiled_points, self._bounds,
                                self._left_slopes, self._right_slopes, self._compiled_slopes, points_is_compiled=True)
            except RuntimeError:
                # If the compiled points and slopes are not on the same device as the input
                self.compile_for_eval()
                return pwlu_forward(norm(x) if norm else x, self._compiled_points, self._bounds,
                                self._left_slopes, self._right_slopes, self._compiled_slopes, points_is_compiled=True)

    def __repr__(self):
        ret = f'{self.__class__}(n_regions={self._n_regions}, bound={self._bounds})'
        if self.channelwise:
            ret += f'n_channels={self._n_channels}'
        return ret

    def get_plottable(self, n_points: int = 100, bound: float = 4) -> torch.Tensor:
        '''
        :param n_points: number of points to plot
        :param bound: bound to plot, defaults to self.bound
        '''

        locs = np.linspace(-bound, bound, n_points)[np.newaxis, np.newaxis, :]
        if self.channelwise:
            locs = locs.repeat(self._n_channels, 1)

        locs = torch.from_numpy(locs)

        try:
            vals = self.forward(locs.cuda(), normed=False)
        except:
            vals = self.forward(locs.cpu(), normed=False)

        vals = vals.detach().cpu().numpy()
        locs = locs.cpu().numpy()
        return locs.squeeze(), vals.squeeze()

    def n_channels(self) -> int:
        return self._n_channels

    def normalize_points(self) -> None:
        pass

    @abstractmethod
    def get_points(self):
        pass

    @abstractmethod
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
    Other parameters are passed to PWLUBase.
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

    def normalize_points(self) -> None:
        normalize(self._points)

    def set_points(self, init) -> None:
        self._init = init
        bound = torch.flatten(self._bounds)[1].item()
        spacing = 2 * bound / self._n_regions

        bound += spacing
        locs = torch.linspace(-bound, bound, self._n_points + 2)
        if self.channelwise:
            locs = locs.repeat(self._n_channels, 1)
        points = init(locs)

        self._left_slopes = Parameter((points[..., 1] - points[..., 0]) / spacing)
        self._right_slopes = Parameter((points[..., -1] - points[..., -2]) / spacing)
        self._points = Parameter(points[..., 1:-1])

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
        bound = torch.flatten(self._bounds)[1].item()
        spacing = 2 * bound / self._n_regions

        bound += spacing
        master_locs = torch.linspace(-bound, bound, self._n_points + 2)
        master_locs = master_locs.repeat(self._n_channels, 1)
        points = init(master_locs)

        self._left_slopes = Parameter((points[..., 1] - points[..., 0]) / spacing)
        self._right_slopes = Parameter((points[..., -1] - points[..., -2]) / spacing)
        self._master_points = Parameter(points[..., 1:-1])
        self._relative_points = Parameter(torch.zeros(self._n_channels, self._n_points))


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
    logging.getLogger().setLevel(logging.DEBUG)
    args = [
        {'n_channels': 10, 'learnable_bound': True, 'same_bound': True},
        {'n_channels': 10, 'learnable_bound': False, 'same_bound': False},
        {'learnable_bound': False, 'same_bound': False},
        {},
    ]
    for kwargs in args:
        logging.debug(f'Performing test on pwlu with {kwargs}')
        pwlu = PWLU(**kwargs)
        assert pwlu(torch.zeros(3, 10, 17, 19)).shape == torch.Size([3, 10, 17, 19])
        pwlu.get_plottable()
    
    logging.debug(f'Performing test on RegularizedPWLU')
    pwlu = RegularizedPWLU(10, learnable_bound=True)
    pwlu.get_plottable()
    assert pwlu(torch.zeros(3, 10, 17, 19)).shape == torch.Size([3, 10, 17, 19])

    logging.info(f'Performing compile accuracy test')
    x = torch.rand(1, 10, 1, 1)
    pwlu = PWLU(10, learnable_bound=True)
    y = pwlu(x)
    pwlu = PWLU(10, learnable_bound=True, autocompile=False)
    z = pwlu(x)
    assert torch.allclose(y, z)

    n_reps = 10000

    logging.info(f'Performing compile speed test with {n_reps=}')
    # 96x7x7 is smallest Imagenet layer size
    for n_regions in (2**x for x in range(1, 8)):
        compile_total = 0
        pwlu = PWLU(96, n_regions=n_regions, autocompile=True)
        for i in range(n_reps):
            x = torch.rand(1, 96, 7, 7)
            start = time.perf_counter()
            y = pwlu(x)
            compile_total += time.perf_counter() - start
        compile_avg = compile_total / n_reps

        reg_total = 0

        pwlu = PWLU(96, n_regions=n_regions, autocompile=False)
        for i in range(n_reps):
            x = torch.rand(1, 96, 7, 7)
            start = time.perf_counter()
            y = pwlu(x)
            reg_total += time.perf_counter() - start
        reg_avg = reg_total / n_reps

        logging.info(
            f'{n_regions=}. Average compiled time: {compile_avg * 10 ** 6:.2f}μs | Average not compiled time: {reg_avg * 10 ** 6:.2f}μs')
