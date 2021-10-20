"""
Daniel Monroe, 2021
Implementation of paper
"Learning specialized activation functions with the Piecewise Linear Unit"
https://arxiv.org/pdf/2104.03693.pdf
The structure is exactly the same as the original, but the implementation has some extra optimizations.
"""

from utils import *
import torch
from torch.nn import Parameter
import torch.nn as nn
from abc import abstractmethod, ABC
import numpy as np
import time
import logging
import numbers

__all__ = ['PWLU', 'PWLUBase', 'RegularizedPWLU', 'normed_pwlu']


def _pwlu_forward(x: torch.Tensor, points: torch.Tensor, left_bounds: torch.Tensor, right_bounds: torch.Tensor,
                  left_diffs: torch.Tensor,
                  right_diffs: torch.Tensor, compiled_diffs: torch.Tensor = None,
                  points_is_compiled: bool = False) -> torch.Tensor:
    """
    Apply PWLU activation function to x
    :param x: input
    :param points: points to evaluate PWLU at
    :param left_bounds: left bounds of PWLU
    :param right_bounds: right bounds of PWLU
    :param left_diffs: left diffs of PWLU
    :param right_diffs: right diffs of PWLU
    :param compiled_diffs: compiled diffs of PWLU
    :param points_is_compiled: if true, points is pre-compiled, i.e., as false_points
    """
    # Initialize variables
    batch_size, n_channels, *other_dims = x.shape
    channelwise = len(points.shape) == 2

    # If points are compiled then they contain the simulated left points
    n_points = points.shape[-1] - 1 if points_is_compiled else points.shape[-1]
    n_regions = n_points - 1

    region_lengths = (right_bounds - left_bounds) / n_regions

    # Compute diffs
    if compiled_diffs is not None:
        diffs = compiled_diffs
    else:
        diffs = (points - torch.roll(points, 1, dims=-1))[..., 1:]
        diffs = torch.cat([left_diffs.unsqueeze(-1), diffs, right_diffs.unsqueeze(-1)], dim=-1)

    # Create normalized version of x; values for bounds will be mapped to 0 (sim_left_bound)
    # and n_regions + 1 (right bound)Some will lie outside

    sim_left_bounds = left_bounds - region_lengths
    if channelwise:
        if not isinstance(region_lengths, numbers.Real):
            shape = (1, -1) + tuple(1 for _ in other_dims)
            sim_left_bounds_reshaped = sim_left_bounds.reshape(shape)
            region_lengths_reshaped = region_lengths.reshape(shape)
        else:
            sim_left_bounds_reshaped = sim_left_bounds
            region_lengths_reshaped = region_lengths
        x_normal = (x - sim_left_bounds_reshaped) / region_lengths_reshaped
    else:
        x_normal = (x - sim_left_bounds) / region_lengths

    # Regions are 0, 1... n_regions, n_regions + 1; outermost are out of bounds
    regions = (x_normal.clamp(0, (n_regions + 1) * 1.001)).long()

    # Create tensor of dists from 0 to 1 from left point: shape (channels, ...) if channelwise else (...)
    dists = x_normal - regions

    # Create tensors of points including simulated left point
    if points_is_compiled:
        false_points = points
    else:
        false_points = torch.cat([(points[..., 0] - left_diffs).unsqueeze(-1), points], dim=-1)

    # Pack regions into form suitable for evaluation
    if channelwise:
        regions_packed = regions
        shape = (1, -1) + tuple(1 for _ in other_dims)
        offsets = torch.arange(n_channels, dtype=torch.long, device=x.device).reshape(*shape) * (n_points + 1)
        regions_packed += offsets

        # Compute left_points and slope necessary for evaluation
        left_points = torch.take(false_points, regions_packed)

        # At this point diffs becomes the gathered diffs
        diffs = torch.take(diffs, regions_packed)

    else:
        regions_packed = regions.reshape(-1)

        # Compute left_points and slope necessary for evaluation
        left_points = torch.gather(false_points, -1, regions_packed)

        # At this point diffs becomes the gathered diffs
        diffs = torch.gather(diffs, -1, regions_packed)

    # Compute activations
    if channelwise:
        left_points = left_points.reshape(x.size())
        diffs = diffs.reshape(x.size())
        ret = left_points + dists * diffs
    else:
        left_points = left_points.reshape(x.size())
        diffs = diffs.reshape(x.size())
        ret = left_points + dists * diffs

    return ret


class PWLUBase(torch.nn.Module, ABC):
    """
    Abstract base class for PWLU
    The number of regions represents the number of regions inside the bounds.
    There are always parameters for controlling slopes outside bounds.
    Accepts n_regions, bound, learnable_bound, same_bound, init, norm, and norm_args arguments to be passed in from child class
    """

    def __init__(self,
                 n_regions: int = 6,
                 bound: float = 2.3,
                 learnable_bound: bool = False,
                 same_bound: bool = False,
                 init='relu',
                 normed: bool = False,
                 norm_args: dict = {},
                 autocompile=True,
                 **kwargs):
        """
        :param n_regions: number of regions in the PWLU
        :param bound: bound of the PWLU
        :param learnable_bound: if true, bound is learnable
        :param same_bound: if true, all channels have the same bound
        :param init: function to initialize the points
        :param normed: whether to add a LazyBatchNorm2d
        :param norm_args: arguments to pass to norm
        :param autocompile: if true, compile the forward function
        :param kwargs: other arguments, will raise exception if not empty
        """

        super().__init__()

        if kwargs:
            raise ValueError(f'Unused keyword arguments: {kwargs}')

        assert hasattr(self, '_n_channels'), 'PWLUBase subclass must set _n_channels before calling super().__init__'

        if learnable_bound:
            if self.channelwise and not same_bound:
                bound = torch.Tensor([[-bound, bound]])
                bound = bound.repeat(self._n_channels, 1)
            else:
                bound = torch.Tensor([-bound, bound])
            self._left_bounds = Parameter(bound[..., 0])
            self._right_bounds = Parameter(bound[..., 1])
        else:
            self._left_bounds = -bound
            self._right_bounds = bound

        self._n_regions = n_regions
        self._n_points = n_regions + 1
        self.set_points(get_activation(init))

        # Compile settings
        self._autocompile = autocompile
        self._compiled_points = None
        self._compiled_diffs = None
        self._compiled = False

        # Create normalization layer
        self._norm = nn.LazyBatchNorm2d(affine=False, **norm_args) if normed else None

    def compile_for_eval(self):
        """
        Compile the points and slopes for evaluation
        The compiled points includes the effective point left of the left bound
        """
        points = self.get_points()
        self._compiled_points = torch.cat([(points[..., 0] - self._left_diffs).unsqueeze(-1), points],
                                          dim=-1).detach().to(points.device)

        diffs = (points - torch.roll(points, 1, dims=-1))[..., 1:]
        diffs = torch.cat([self._left_diffs.unsqueeze(-1), diffs, self._right_diffs.unsqueeze(-1)], dim=-1)
        self._compiled_diffs = diffs.detach().to(points.device)

        self._compiled = True

    def forward(self, x: torch.Tensor, normed: bool = True):
        """
        Forward function of the BasePWLU
        :param x: input tensor
        :param normed: if true, use self._norm on the input
        """
        norm = self._norm if normed else None
        if self.training or not self._autocompile:
            # Compile as if in training mode
            self._compiled = False
            out = _pwlu_forward(norm(x) if norm else x, self.get_points(), self._left_bounds, self._right_bounds,
                                self._left_diffs,
                                self._right_diffs)

            return out
        else:
            # Compile points and slopes if not compiled, run using compiled form
            if not self._compiled:
                self.compile_for_eval()
            return _pwlu_forward(norm(x) if norm else x, self._compiled_points, self._left_bounds,
                                 self._right_bounds,
                                 self._left_diffs, self._right_diffs, self._compiled_diffs,
                                 points_is_compiled=True)


    def __repr__(self):
        ret = f'{self.__class__}(n_regions={self._n_regions}, bound={self._bounds})'
        if self.channelwise:
            ret += f'n_channels={self._n_channels}'
        return ret

    def get_plottable(self, n_points: int = 100, bound: float = 3) -> torch.Tensor:
        """
        :param n_points: number of points to plot
        :param bound: bound to plot, defaults to self.bound
        """

        locs = np.linspace(-bound, bound, n_points)[np.newaxis, np.newaxis, :]
        if self.channelwise:
            locs = locs.repeat(self._n_channels, 1)

        locs = torch.from_numpy(locs)

        try:
            vals = self(locs.cuda(), normed=False)
        except:
            vals = self(locs.cpu(), normed=False)

        vals = vals.detach().cpu().numpy()
        locs = locs.cpu().numpy()
        return locs.squeeze(), vals.squeeze()

    def normalize_points(self) -> None:
        pass

    @abstractmethod
    def get_points(self):
        pass

    @abstractmethod
    def set_points(self):
        pass

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def channelwise(self) -> bool:
        return self._n_channels is not None

    @property
    def n_regions(self) -> int:
        return self._n_regions

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def autocompile(self) -> bool:
        return self._autocompile

    @autocompile.setter
    def autocompile(self, value: bool):
        assert isinstance(value, bool)
        self._autocompile = value

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
    """
    Vanilla implementation of the PWLU.
    The PWLU is initialized as channelwise if n_channels is specified; otherwise, it is layerwise.
    Other parameters are passed to PWLUBase.
    """

    def __init__(self,
                 n_channels: int = None,
                 **kwargs):
        """
        :param n_channels: number of channels in input, defaults to None
        :param kwards: see PWLUBase
        """
        self._n_channels = n_channels
        super().__init__(**kwargs)
        self.set_points(self._init)

    def normalize_points(self) -> None:
        factor = normalize(self._points)
        with torch.no_grad():
            self._left_diffs /= factor
            self._right_diffs /= factor

    def set_points(self, init) -> None:
        self._init = init
        bound = self._right_bounds if isinstance(self._right_bounds, numbers.Real) else torch.flatten(self._right_bounds)[
            0].item()
        spacing = 2 * bound / self._n_regions

        bound += spacing
        locs = torch.linspace(-bound, bound, self._n_points + 2)
        if self.channelwise:
            locs = locs.repeat(self._n_channels, 1)
        points = init(locs)

        self._left_diffs = Parameter(points[..., 1] - points[..., 0])
        self._right_diffs = Parameter(points[..., -1] - points[..., -2])
        self._points = Parameter(points[..., 1:-1])

        self.normalize_points()

    def get_points(self) -> torch.Tensor:
        return self._points

    @property
    def points(self):
        return self._points


class RegularizedPWLU(PWLUBase):
    """
    The regularized PWLU is a a stable version of the channelwise PWLU meant for smaller networks.
    It is channelwise by definition and has a "master points" parameter which is the average of the other channel units
    and "deviation points" parameters which are the deviations from the master points, controlled in their strength by the relative_factor parameter.
    Other parameters are passed to PWLUBase.
    """

    def normalize_points(self) -> None:
        factor = normalize(self._master_points)
        normalize(self._relative_points, fix_std=False)
        with torch.no_grad():
            self._left_diffs /= factor
            self._right_diffs /= factor

    def __init__(self,
                 n_channels: int,
                 *,
                 relative_factor=0.5,
                 **kwargs):
        """
        :param n_channels: number of channels in input
        :param relative_factor: how much to scale the relative point deviations by
        :param kwargs: see PWLUBase
        """

        self._n_channels = n_channels
        super().__init__(same_bound=True, **kwargs)
        self._relative_factor = relative_factor
        self._master_points = Parameter(torch.zeros(self._n_points))
        self._relative_points = Parameter(torch.zeros(self._n_channels, self._n_points))
        self.set_points(self._init)

    def get_points(self) -> torch.Tensor:
        return self._master_points + self._relative_points * self._relative_factor

    def set_points(self, init) -> None:
        self._init = init
        bound = self._right_bounds if isinstance(self._right_bounds, numbers.Real) else torch.flatten(self._right_bounds)[
            0].item()
        spacing = 2 * bound / self._n_regions

        bound += spacing
        master_locs = torch.linspace(-bound, bound, self._n_points + 2)
        master_locs = master_locs.repeat(self._n_channels, 1)
        points = init(master_locs)

        self._left_diffs = Parameter(points[..., 1] - points[..., 0])
        self._right_diffs = Parameter(points[..., -1] - points[..., -2])
        self._master_points = Parameter(points[..., 1:-1])
        self._relative_points = Parameter(torch.zeros(self._n_channels, self._n_points))
        self.normalize_points()


def normed_pwlu(pwlu_class: type, *args, norm: nn.Module, norm_args: dict = {}, **kwargs) -> PWLUBase:
    """
    Make a PWLU with a normalization layer.
    :param pwlu_class: PWLU class to instantiate
    :param args: positional arguments to pass to PWLU class, usually just num channels
    :param norm: normalization layer
    :param norm_args: arguments to pass to normalization layer
    :param kwargs: keyword arguments to pass to PWLU class
    """
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
        pwlu.training=False
        assert pwlu(torch.zeros(3, 10, 17, 19)).shape == torch.Size([3, 10, 17, 19])
        pwlu.get_plottable()

    logging.debug(f'Performing test on RegularizedPWLU')
    pwlu = RegularizedPWLU(10, learnable_bound=True)
    pwlu.get_plottable()
    assert pwlu(torch.zeros(3, 10, 17, 19)).shape == torch.Size([3, 10, 17, 19])

    logging.info(f'Performing compile accuracy test')
    x = torch.rand(1, 10, 1, 1)
    pwlu = PWLU(10, learnable_bound=True)
    pwlu.training=True
    y = pwlu(x)
    pwlu = PWLU(10, learnable_bound=True, autocompile=False)
    z = pwlu(x)
    assert torch.allclose(y, z)

    n_reps = 200

    logging.info(f'Performing n_regions speed test with {n_reps=}')
    # 96x7x7 is smallest Imagenet layer size
    many_regions_total = 0
    pwlu = PWLU(96, n_regions=1000)
    pwlu.training=True
    for i in range(n_reps):
        x = torch.rand(64, 96, 7, 7)
        start = time.perf_counter()
        y = pwlu(x)
        y.sum().backward()
        many_regions_total += time.perf_counter() - start
    many_regions_avg = many_regions_total / n_reps

    few_regions_total = 0
    pwlu = PWLU(96, n_regions=4)
    pwlu.training = True
    for i in range(n_reps):
        x = torch.rand(64, 96, 7, 7)
        start = time.perf_counter()
        y = pwlu(x)
        y.sum().backward()
        few_regions_total += time.perf_counter() - start
    few_regions_avg = few_regions_total / n_reps

    logging.info(
        f'Average many regions time: {round(many_regions_avg * 10 ** 6)}μs |'
        f' Average few regions time: {round(few_regions_avg * 10 ** 6)}μs')

