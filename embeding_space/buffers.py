from stable_baselines3.common.buffers import RolloutBuffer, BaseBuffer
import numpy as np
from typing import Generator, Optional, Union
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

import torch as th
from gym import spaces
from stable_baselines3.common.vec_env import VecNormalize

class EmbedingRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_probs: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    rewards: th.Tensor
    entropys: th.Tensor
    inference_log_probs: th.Tensor
    zs: th.Tensor
    embeding_entropys: th.Tensor
    obs_add_zs: th.Tensor
    gammas: th.Tensor


class EmbedingRolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        embeding_dim: int,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.embeding_dim = embeding_dim
        self.reset()
        
    def reset(self) -> None:

        self.observations = th.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=th.float32)
        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32)
        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.returns = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.dones = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.values = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.advantages = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.generator_ready = False

        self.entropys = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.inference_log_probs = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.zs = th.zeros((self.buffer_size, self.n_envs, self.embeding_dim), dtype=th.float32)
        self.embeding_entropys = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.obs_add_zs = th.zeros((self.buffer_size, self.n_envs, self.obs_shape[0]+self.embeding_dim), dtype=th.float32)
        self.gammas = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)

        super().reset()

    def add(
        self, obs: th.Tensor, action: th.Tensor, reward: th.Tensor, done: th.Tensor, value: th.Tensor, log_prob: th.Tensor, 
        entropy: th.Tensor, inference_log_prob: th.Tensor, z: th.Tensor, embeding_entropy: th.Tensor, obs_add_z: th.Tensor
    ) -> None:
        """
        :param obs: (np.ndarray) Observation
        :param action: (np.ndarray) Action
        :param reward: (np.ndarray)
        :param done: (np.ndarray) End of episode signal.
        :param value: (th.Tensor) estimated value of the current state
            following the current policy.
        :param log_prob: (th.Tensor) log probability of the action
            following the current policy.
        """
        self.observations[self.pos] = (obs).clone()
        self.actions[self.pos] = (action).clone()
        self.rewards[self.pos] = (reward).clone()
        self.dones[self.pos] = (done).clone()
        self.values[self.pos] = value.clone().flatten()
        self.log_probs[self.pos] = log_prob.clone()
        self.entropys[self.pos] = (entropy).clone()
        self.inference_log_probs[self.pos] = (inference_log_prob).clone()
        self.zs[self.pos] = (z).clone()
        self.embeding_entropys[self.pos] = (embeding_entropy).clone()
        self.obs_add_zs[self.pos] = (obs_add_z).clone()
        self.gammas[self.pos] = th.tensor(([self.gamma**self.pos]*self.n_envs))

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[EmbedingRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ["observations", "actions", "values", "log_probs", "advantages", "returns", 
                           "rewards", "entropys", "inference_log_probs", "zs", "embeding_entropys", "obs_add_zs", "gammas"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> EmbedingRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.rewards[batch_inds].flatten(),
            self.entropys[batch_inds].flatten(),
            self.inference_log_probs[batch_inds].flatten(),
            self.zs[batch_inds].flatten(),
            self.embeding_entropys[batch_inds].flatten(),
            self.obs_add_zs[batch_inds].flatten(),
            self.gammas[batch_inds].flatten()
        )
        return EmbedingRolloutBufferSamples(*data)

    def swap_and_flatten(self, arr: th.Tensor) -> th.Tensor:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr: (np.ndarray)
        :return: (np.ndarray)
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
            ret = arr.permute(1, 0).reshape(shape[0] * shape[1], *shape[2:])
        else:
            ret = arr.permute(1, 0, 2).reshape(shape[0] * shape[1], *shape[2:])

        return ret




