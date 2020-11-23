from typing import Any, Dict, List, Optional, Tuple
import gym
from gym import spaces

from torch.distributions import Bernoulli, Categorical, Normal
from torch.distributions.utils import _standard_normal, broadcast_all
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.preprocessing import get_action_dim

class ReproduceNormal(Normal):
    def rsample2(self, sample_shape=th.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale, eps

    def reproduce_sample(self, eps):
        return self.loc + eps * self.scale

class ReproduceDiagGaussianDistribution(DiagGaussianDistribution):
    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions: (th.Tensor)
        :param log_std: (th.Tensor)
        :return: (DiagGaussianDistribution)
        """
        #action_std = th.ones_like(mean_actions) * log_std.exp()
        action_std = th.ones_like(mean_actions) * 0.2
        self.distribution = ReproduceNormal(mean_actions, action_std)
        return self

    def sample2(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample2()

    def reproduce_sample(self, eps):
        return self.distribution.reproduce_sample(eps)

    def get_actions2(self) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic: (bool)
        :return: (th.Tensor)
        """
        return self.sample2()


def make_proba_distribution(
    action_space: gym.spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: (gym.spaces.Space) the input action space
    :param use_sde: (bool) Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: (Optional[Dict[str, Any]]) Keyword arguments to pass to the probability distribution
    :return: (Distribution) the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        assert len(action_space.shape) == 1, "Error: the action space must be a vector"
        cls = StateDependentNoiseDistribution if use_sde else ReproduceDiagGaussianDistribution
        return cls(get_action_dim(action_space), **dist_kwargs)
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(action_space.n, **dist_kwargs)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(action_space.nvec, **dist_kwargs)
    elif isinstance(action_space, spaces.MultiBinary):
        return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )
