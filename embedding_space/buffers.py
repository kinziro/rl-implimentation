from stable_baselines3.common.buffers import RolloutBuffer, BaseBuffer
import numpy as np
from typing import Generator, Optional, Union
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

import torch as th
from gym import spaces
from stable_baselines3.common.vec_env import VecNormalize

class EmbeddingRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    cut_entropys_before_z: th.Tensor
    cut_log_probs_before_z: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    rewards: th.Tensor
    log_probs: th.Tensor
    entropys: th.Tensor
    inference_log_probs: th.Tensor
    cut_inference_log_probs_before_z: th.Tensor
    cut_inference_log_probs_before_action: th.Tensor
    zs: th.Tensor
    embedding_log_probs: th.Tensor
    embedding_entropys: th.Tensor
    obs_add_zs: th.Tensor
    gammas: th.Tensor
    error_zs: th.Tensor

class EmbeddingRolloutBufferSamples2(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    cut_entropys_before_z: th.Tensor
    cut_log_probs_before_z: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    rewards: th.Tensor
    log_probs: th.Tensor
    entropys: th.Tensor
    inference_log_probs: th.Tensor
    cut_inference_log_probs_before_z: th.Tensor
    cut_inference_log_probs_before_action: th.Tensor
    zs: th.Tensor
    embedding_log_probs: th.Tensor
    embedding_entropys: th.Tensor
    obs_add_zs: th.Tensor
    gammas: th.Tensor
    error_zs: th.Tensor
    data_lens: th.Tensor


class EmbeddingRolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        embedding_dim: int,
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
        self.embedding_dim = embedding_dim
        self.variables = ["observations", "actions", "cut_entropys_before_z", "cut_log_probs_before_z", "advantages", "returns", 
                          "rewards", "log_probs","entropys", "inference_log_probs", "cut_inference_log_probs_before_z", 
                          "cut_inference_log_probs_before_action", "zs", "embedding_log_probs", "embedding_entropys", "obs_add_zs", 
                          "gammas", "error_zs"]
        self.reset()
        
    def reset(self) -> None:

        self.observations = th.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=th.float32)
        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32)
        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.returns = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.dones = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.cut_entropys_before_z = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.cut_log_probs_before_z = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.advantages = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.generator_ready = False

        self.log_probs = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.entropys = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.inference_log_probs = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.cut_inference_log_probs_before_z = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.cut_inference_log_probs_before_action = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.zs = th.zeros((self.buffer_size, self.n_envs, self.embedding_dim), dtype=th.float32)
        self.embedding_log_probs = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.embedding_entropys = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.obs_add_zs = th.zeros((self.buffer_size, self.n_envs, self.obs_shape[0]+self.embedding_dim), dtype=th.float32)
        self.gammas = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.error_zs = th.zeros((self.buffer_size, self.n_envs, self.embedding_dim), dtype=th.float32)

        super().reset()

    def add(
        self, obs: th.Tensor, action: th.Tensor, reward: th.Tensor, done: th.Tensor, cut_entropy_before_z: th.Tensor, 
        cut_log_prob_before_z: th.Tensor, log_prob, entropy: th.Tensor, inference_log_prob: th.Tensor, cut_inference_log_prob_before_z: th.Tensor, 
        cut_inference_log_prob_before_action: th.Tensor, z: th.Tensor, embedding_log_prob: th.Tensor, embedding_entropy: th.Tensor, 
        obs_add_z: th.Tensor, error_z: th.Tensor
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
        self.cut_entropys_before_z[self.pos] = cut_entropy_before_z.clone().flatten()
        self.cut_log_probs_before_z[self.pos] = cut_log_prob_before_z.clone()
        self.log_probs[self.pos] = log_prob.clone()
        self.entropys[self.pos] = (entropy).clone()
        self.inference_log_probs[self.pos] = (inference_log_prob).clone()
        self.cut_inference_log_probs_before_z[self.pos] = cut_inference_log_prob_before_z.clone()
        self.cut_inference_log_probs_before_action[self.pos] = cut_inference_log_prob_before_action.clone()
        self.zs[self.pos] = (z).clone()
        self.embedding_log_probs[self.pos] = (embedding_log_prob).clone()
        self.embedding_entropys[self.pos] = (embedding_entropy).clone()
        self.obs_add_zs[self.pos] = (obs_add_z).clone()
        self.gammas[self.pos] = th.tensor(([self.gamma**self.pos]*self.n_envs))
        self.error_zs[self.pos] = error_z.clone()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[EmbeddingRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in self.variables:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        
        self.observations.shape
        self.rewards.shape

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> EmbeddingRolloutBufferSamples:
        data = (
            self.observations,
            self.actions,
            self.cut_entropys_before_z,
            self.cut_log_probs_before_z,
            self.advantages,
            self.returns,
            self.rewards,
            self.log_probs,
            self.entropys,
            self.inference_log_probs,
            self.cut_inference_log_probs_before_z,
            self.cut_inference_log_probs_before_action,
            self.zs,
            self.embedding_log_probs,
            self.embedding_entropys,
            self.obs_add_zs,
            self.gammas,
            self.error_zs,
        )
        return EmbeddingRolloutBufferSamples(*data)

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
            #ret = arr.permute(1, 0).reshape(shape[0] * shape[1], *shape[2:])
            ret = arr.permute(1, 0)
        else:
            #ret = arr.permute(1, 0, 2).reshape(shape[0] * shape[1], *shape[2:])
            ret = arr.permute(1, 0, 2)

        return ret

    def get_each_cycle(self):
        self.dones = self.swap_and_flatten(self.dones)
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in self.variables:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True
        
        # 各環境の探索サイクルの区切りを計算
        separations = []
        dones_np = self.dones.detach().numpy()
        indexes = np.array(list(range(self.buffer_size+1))).reshape(1, -1)
        for e in range(self.n_envs):
            dones_e = dones_np[e, :].reshape(1, -1)
            dones_e = np.hstack([np.array([[1]]), dones_e])
            separation_e = list(indexes[np.where(dones_e==1)])
            separations.append(separation_e)

        data_list = []
        data_lens = []
        for var_i, var in enumerate(self.variables):
            tar_var =  self.__dict__[var]
            if len(tar_var.shape) == 2:
                v = th.zeros((self.buffer_size*self.n_envs, self.buffer_size), dtype=th.float32)
            else:
                v = th.zeros((self.buffer_size*self.n_envs, self.buffer_size, tar_var.shape[2]), dtype=th.float32)
            
            v_idx = 0
            for e in range(self.n_envs):
                for i in range(len(separations[e])-1):
                    str_idx = separations[e][i]
                    end_idx = separations[e][i+1]
                    data_len = end_idx - str_idx
                    v[v_idx, 0:data_len] = tar_var[e, str_idx:end_idx]

                    if var_i == 0:
                        data_lens.append(data_len)
                    
                    v_idx += 1
            
            if var == "gammas":
                v[0:v_idx] = v[0:v_idx] / v[0:v_idx, 0].reshape(-1, 1)

            data_list.append(v[0:v_idx])
        
        data_list.append(th.tensor(data_lens))


        yield EmbeddingRolloutBufferSamples2(*data_list)

                   

                

