import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
from gym import spaces
import numpy as np
import torch as th

from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv

from network import InferenceNet, EmbeddingNet, PolicyNet
from buffers import EmbeddingRolloutBuffer

import copy

class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: (int) Seed for the pseudo random generators
    :param device: (str or th.device) Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        task_id_list: list = None,
        embedding_dim: int = 3,
        loss_alphas: list = [1, 1, 1]
    ):

        super(OnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=PolicyNet,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )

        self.device_str = device
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        if self.env is not None:
            self.env_task_id_list = [e.env.task_id for e in self.env.envs]
        self.task_id_list = task_id_list
        if self.task_id_list is not None:
            self.task_id_dim = len(self.task_id_list[0])
        self.embedding_dim = embedding_dim

        self.inference_kwargs = inference_kwargs
        self.embedding_kwargs = embedding_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        if _init_setup_model:
            self._setup_model()
        
        self.alpha_1, self.alpha_2, self.alpha_3 = loss_alphas

        if self.inference_kwargs is not None:
            self.n_obs_history = inference_kwargs["n_obs_history"]
        else:
            self.n_obs_history = 1
        if self.env is not None:
            self.obs_h_manager = ObservationHistoryManager(n_envs=self.env.num_envs, n_history=self.n_obs_history)
            self.obs_h_manager.add(self.env.reset())        # 他でもenv.reset()がされていないか要確認。環境の観測値の初期値にランダム要素がある場合バグになる可能性あり。
        
        self.task_id_idx = 0
        
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        policy_observation_dim = self.observation_space.shape[0] + self.embedding_dim
        policy_observation_high = max(self.observation_space.high)
        observation_high = np.array([policy_observation_high] * policy_observation_dim)
        policy_observation_space = spaces.Box(-observation_high, observation_high)
        policy_normalization_coeff = [self.observation_space.high[0]]*self.observation_space.shape[0] + [1]*self.embedding_dim

        self.rollout_buffer = EmbeddingRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.embedding_dim,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )


        # Policyの作成
        # zを観測値に追加
        self.policy = self.policy_class(
            policy_observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            device=self.device,
            normalization_coeff=None,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # 埋め込みネットワークの作成
        emb_normalization_coeff = [1]*self.task_id_dim
        embedding_kwargs = {} if self.embedding_kwargs is None else self.embedding_kwargs
        self.embedding_net = EmbeddingNet(task_id_dim=self.task_id_dim, embedding_dim=self.embedding_dim, 
                                          normalization_coeff=None, device=self.device, **embedding_kwargs).to(self.device)
        self.embedding_optimizer = th.optim.RMSprop(self.embedding_net.parameters(), **self.optimizer_kwargs)

        # 推論ネットワークの作成
        obs_high = float(self.observation_space.high[0])
        act_high =  float(self.action_space.high[0])
        inf_normalization_coeff = [obs_high] * self.observation_space.shape[0] + [act_high] * self.action_space.shape[0]
        #inf_obs_len = self.observation_space.shape[0]*self.n_obs_history
        inference_kwargs = {} if self.inference_kwargs is None else self.inference_kwargs
        self.inference_net = InferenceNet(observation_space=self.observation_space.shape[0], action_space=self.action_space.shape[0], 
                                          embedding_dim=self.embedding_dim, normalization_coeff=None, 
                                          device=self.device, **inference_kwargs).to(self.device)
        self.inference_optimizer = th.optim.RMSprop(self.inference_net.parameters(), **self.optimizer_kwargs)

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.

        :param env: (VecEnv) The training environment
        :param callback: (BaseCallback) Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: (RolloutBuffer) Buffer to fill with rollouts
        :param n_steps: (int) Number of experiences to collect per environment
        :return: (bool) True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        #for i in range(self.n_envs):
        #    self.env.envs[i].env.set_goal(self.task_id_list[self.task_id_idx])
        self._last_obs = self.env.reset()
        self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool)
        self.obs_h_manager = ObservationHistoryManager(n_envs=self.env.num_envs, n_history=self.n_obs_history)
        self.obs_h_manager.add(self._last_obs)

        #self.task_id_idx = (self.task_id_idx + 1) % len(self.task_id_list)

        # zは更新ごとに一回だけサンプリング
        # 潜在変数を推定
        t = th.tensor(self.env_task_id_list).float().to(self.device)
        z, embedding_log_prob, embedding_entropy = self.embedding_net.forward(t)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            #-- embedding_net更新用 --
            # 潜在変数を観測値に追加
            obs = self._last_obs
            obs_h = self.obs_h_manager.get()

            obs_add_z_tensor = th.cat([th.as_tensor(obs), z], dim=1)
            actions, _, log_prob, entropy = self.policy.forward(obs_add_z_tensor)

            # 推論NNによる潜在変数を取得
            obs_add_action_tensor = th.cat([th.tensor(obs_h), actions], dim=1)
            _, inference_log_prob = self.inference_net(obs_add_action_tensor)
            # embedding_netとinference_netのzの差を計算
            with th.no_grad():
                z_mean, _, _ = self.embedding_net.predict(t)
                inference_z_mean, _ = self.inference_net.predict(obs_add_action_tensor)
                error_z = th.abs(z_mean - inference_z_mean)

            #-- policy_net更新用 --
            # 微分の際にembedding_netの影響を受けないために、z以前の連鎖をカット
            with th.no_grad():
                cut_obs_add_z_before_z = th.tensor(obs_add_z_tensor.detach().numpy())
            cut_action_before_z, _, cut_log_probs_before_z, cut_entropys_before_z = self.policy.forward(cut_obs_add_z_before_z, reproduce=True)
            # 推論NNによる潜在変数を取得
            cut_obs_add_action_before_z = th.cat([th.tensor(obs_h), cut_action_before_z], dim=1)
            _, cut_inference_log_prob_before_z = self.inference_net(cut_obs_add_action_before_z, reproduce=True)

            #-- inference_net更新用 --
            # 微分の際にembedding_net, policy_netの影響を受けないために、action以前の連鎖をカット
            with th.no_grad():
                cut_obs_add_action_before_action = th.tensor(cut_obs_add_action_before_z.detach().numpy())
            _, cut_inference_log_prob_before_action = self.inference_net(cut_obs_add_action_before_action, reproduce=True)

            #actions = actions.cpu().numpy()
            actions = actions.detach().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if callback.on_step() is False:
                print('callback on step')
                return False

            self._update_info_buffer(infos)
            n_steps += 1
            self.num_timesteps += env.num_envs

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(th.tensor(self._last_obs), th.tensor(actions), th.tensor(rewards), th.tensor(self._last_dones), 
                               cut_entropys_before_z, cut_log_probs_before_z, log_prob, entropy, inference_log_prob,  cut_inference_log_prob_before_z,
                               cut_inference_log_prob_before_action, z, embedding_log_prob, embedding_entropy, obs_add_z_tensor, error_z)
            self._last_obs = new_obs
            self._last_dones = dones
            self.obs_h_manager.reset(dones)
            self.obs_h_manager.add(new_obs)

        #rollout_buffer.compute_returns_and_advantage(values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded by default
        when saving the model.

        :return: ([str]) List of parameters that should be excluded from save
        """
        return ["policy", "embedding_net", "inference_net", "embedding_optimizer", "inference_optimizer", 
                "device", "env", "eval_env", "replay_buffer", "rollout_buffer", "_vec_normalize_env"]

    def get_torch_variables(self) -> Tuple[List[str], List[str]]:
        """
        Get the name of the torch variables that will be saved.
        ``th.save`` and ``th.load`` will be used with the right device
        instead of the default pickling strategy.

        :return: (Tuple[List[str], List[str]])
            name of the variables with state dicts to save, name of additional torch tensors,
        """
        state_dicts = ["policy", "embedding_net", "inference_net", "embedding_optimizer", "inference_optimizer"]

        return state_dicts, []


class ObservationHistory:
    def __init__(self, n_history: int):
        self.n_history = n_history
        self.obs_h = [[] for _ in range(n_history)]
        self.index = -1
        self.reset_flag = True
    
    def add(self, obs: list):
        self.index += 1
        self.index %= self.n_history

        if self.reset_flag:
            self.obs_h = [copy.deepcopy(obs) for _ in range(self.n_history)]
            self.reset_flag = False
        else:
            self.obs_h[self.index] = copy.deepcopy(obs)

    def get(self) -> list:
        ret_obs_h = []
        [ret_obs_h.extend(self.obs_h[self.index-i]) for i in range(self.n_history)]
        return ret_obs_h
    
    def reset(self):
        self.obs_h = [[] for _ in range(self.n_history)]
        self.reset_flag = True


class ObservationHistoryManager:
    def __init__(self, n_envs, n_history):
        self.n_envs = n_envs
        self.envs = [ObservationHistory(n_history) for _ in range(n_envs)]
    
    def add(self, obs_np: np.ndarray):
        for i in range(self.n_envs):
            self.envs[i].add(obs_np[i, :].tolist())
    
    def get(self) -> np.ndarray:
        ret = [self.envs[i].get() for i in range(self.n_envs)]
        ret_np = np.array(ret, dtype=np.float32)

        return ret_np
    
    def reset(self, dones: np.ndarray):
        for flag, env in zip(dones, self.envs):
            if flag:
                env.reset()


if __name__=="__main__":
    obs_h_m = ObservationHistoryManager(3, 4)
    print(obs_h_m.get())
    a = np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1]])
    obs_h_m.add(a)
    print(obs_h_m.get())
    a = np.array([[1, 2, 2], [2, 2, 2], [3, 2, 2]])
    obs_h_m.add(a)
    print(obs_h_m.get())
    a = np.array([[1, 3, 3], [2, 3, 3], [3, 3, 3]])
    obs_h_m.add(a)
    print(obs_h_m.get())
    a = np.array([[1, 4, 4], [2, 4, 4], [3, 4, 4]])
    obs_h_m.add(a)
    print(obs_h_m.get())
    a = np.array([[1, 5, 5], [2, 5, 5], [3, 5, 5]])
    obs_h_m.add(a)
    print(obs_h_m.get())
    obs_h_m.reset(np.array([True, False, False]))
    a = np.array([[1, 6, 6], [2, 6, 6], [3, 6, 6]])
    obs_h_m.add(a)
    print(obs_h_m.get())