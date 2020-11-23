from typing import Any, Callable, Dict, Optional, Type, Union

import torch as th
from gym import spaces
from torch.nn import functional as F
import numpy as np

from stable_baselines3.common import logger
#from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import explained_variance

import copy
import pickle
import matplotlib.pyplot as plt

class A2C(OnPolicyAlgorithm):
    """
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param rms_prop_eps: (float) RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: (bool) Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param normalize_advantage: (bool) Whether to normalize or not the advantage
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
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
        learning_rate: Union[float, Callable] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
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

        super(A2C, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            inference_kwargs=inference_kwargs,
            embedding_kwargs=embedding_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            task_id_list=task_id_list,
            embedding_dim=embedding_dim,
            loss_alphas=loss_alphas
        )

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if optimizer_kwargs is None:
            if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
                self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
                self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)
        else:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = optimizer_kwargs

        if _init_setup_model:
            self._setup_model()
        

        self.policy_w = None
        self.inference_w = None
        self.embeding_w = None
        self.policcy_g = None
        self.inference_g = None
        self.embeding_g = None

        self.i_policy_w = None
        self.i_inference_w = None
        self.i_embeding_w = None
        self.i_policcy_g = None
        self.i_inference_g = None
        self.i_embeding_g = None

        self.policy_loss_history = []
        self.inference_loss_history = []
        self.embedding_loss_history = []
        self.reward_history = []
        self.R_history = []
        self.error_zs_history = []

        self.policy_g_history = []
        self.inference_g_history = []
        self.embedding_g_history = []

        self.policy_w_history = []
        self.inference_w_history = []
        self.embedding_w_history = []

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        self.policy.train()
        self.inference_net.train()
        self.embedding_net.train()
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        #a = self.rollout_buffer.get_each_cycle()
        #for rollout_data in self.rollout_buffer.get(batch_size=None):
        for rollout_data in self.rollout_buffer.get_each_cycle():

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            # data from rollbuffer
            cut_log_probs_before_z = rollout_data.cut_log_probs_before_z
            cut_entropys_before_z = rollout_data.cut_entropys_before_z
            log_probs = rollout_data.log_probs
            entropys = rollout_data.entropys
            actions = rollout_data.actions
            rewards = rollout_data.rewards
            inference_log_probs = rollout_data.inference_log_probs
            cut_inference_log_probs_before_z = rollout_data.cut_inference_log_probs_before_z
            cut_inference_log_probs_before_action = rollout_data.cut_inference_log_probs_before_action
            embedding_entropys = rollout_data.embedding_entropys
            gammas = rollout_data.gammas
            data_lens = rollout_data.data_lens
            origin_env_idx = rollout_data.origin_env_idx
            env_weight = rollout_data.env_weight

            with th.no_grad():
                gamma_r_hats = gammas*(rewards + self.alpha_2*inference_log_probs + self.alpha_3*entropys)
                R = gamma_r_hats.sum(dim=1)
                R = th.tensor(R.detach().numpy())    # 勾配計算の影響をなくすために、定数化

                #R_for_h = R / env_weight
                #reward_sum = (gammas*rewards).sum(dim=1) / env_weight
                R_for_h = R
                reward_sum = (gammas*rewards).sum(dim=1)

            # embedding loss
            embedding_log_probs = rollout_data.embedding_log_probs
            embedding_log_probs_mean = embedding_log_probs.sum(dim=1) / data_lens
            embedding_entropys_mean = embedding_entropys.sum(dim=1) / data_lens

            #embedding_entropys = rollout_data.embedding_entropys
            embedding_expected_term_1_each_cpu = R * (log_probs.sum(dim=1) + embedding_log_probs_mean)
            embedding_expected_term_2_each_cpu = (gammas * self.alpha_2 * inference_log_probs + gammas * self.alpha_3 * entropys).sum(dim=1)
            embedding_expected_term_each_cpu = embedding_expected_term_1_each_cpu + embedding_expected_term_2_each_cpu
            embedding_loss_each_cpu = -1 * (embedding_expected_term_each_cpu + self.alpha_1 * embedding_entropys_mean)
            #embedding_loss = (embedding_loss_each_cpu / env_weight).mean()
            embedding_loss = (embedding_loss_each_cpu).mean()

            #a = embedding_loss_each_cpu.detach().numpy()
            #b = embedding_loss_each_cpu_weight.detach().numpy()
            #c = env_weight.detach().numpy()
            #if self.num_timesteps > 5000:
            #    print('')

            # policy loss
            log_prob_sum = R * cut_log_probs_before_z.sum(dim=1)
            inference_log_probs_sum = (gammas * self.alpha_2 * cut_inference_log_probs_before_z).sum(dim=1)
            entropy_sum = (self.alpha_3 * gammas * cut_entropys_before_z).sum(dim=1)

            policy_loss_each_cpu = -1 * (log_prob_sum + inference_log_probs_sum + entropy_sum)
            #policy_loss = (policy_loss_each_cpu / env_weight).mean()
            policy_loss = (policy_loss_each_cpu).mean()

            # inference loss
            inference_loss_each_cpu = -1 * (self.alpha_2 * gammas * cut_inference_log_probs_before_action).sum(dim=1)
            #inference_loss = (inference_loss_each_cpu / env_weight).mean()
            inference_loss = (inference_loss_each_cpu).mean()

            # Optimization step
            # embedding parameter optimization
            self.policy.optimizer.zero_grad()
            self.inference_optimizer.zero_grad()
            self.embedding_optimizer.zero_grad()
            embedding_loss.backward(retain_graph=True)

            policy_g_0 = copy.deepcopy(self.policy.mlp_extractor.shared_net[0].weight.grad.detach().numpy())
            inference_g_0 = copy.deepcopy(self.inference_net.fc_mean.weight.grad.detach().numpy())
            embedding_g_0 = copy.deepcopy(self.embedding_net.fc_mean.weight.grad.detach().numpy())

            policy_w_0 = copy.deepcopy(self.policy.mlp_extractor.shared_net[0].weight.detach().numpy())
            inference_w_0 = copy.deepcopy(self.inference_net.fc_mean.weight.detach().numpy())
            embedding_w_0 = copy.deepcopy(self.embedding_net.fc_mean.weight.detach().numpy())

            self.policy_g_history.append([np.max(policy_g_0), np.min(policy_g_0)])
            self.inference_g_history.append([np.max(inference_g_0), np.min(inference_g_0)])
            self.embedding_g_history.append([np.max(embedding_g_0), np.min(embedding_g_0)])

            self.policy_w_history.append([np.max(policy_w_0), np.min(policy_w_0)])
            self.inference_w_history.append([np.max(inference_w_0), np.min(inference_w_0)])
            self.embedding_w_history.append([np.max(embedding_w_0), np.min(embedding_w_0)])

            import math
            if math.isnan(policy_g_0[0, 0]) or math.isnan(inference_g_0[0, 0]) or math.isnan(embedding_g_0[0, 0]):
                #for data, label in zip([self.policy_g_history, self.inference_g_history, self.embedding_g_history, self.policy_w_history, self.inference_w_history, self.embedding_w_history], 
                #                       ['policy_g', 'inference_g', 'embedding_g', 'policy_w', 'inference_w', 'embedding_w']):
                #    data_np = np.array(data)
                #    fig = plt.figure()
                #    plt.plot(data_np[:, 0], label='max')
                #    plt.plot(data_np[:, 1], label='min')
                #    plt.grid()
                #    plt.legend()
                #    plt.savefig(f'./{label}_max_min.png')
                print('nan occured 1')
            else:
                self.embedding_optimizer.step()

            th.nn.utils.clip_grad_norm_(self.embedding_net.parameters(), self.max_grad_norm)

            #policy_g_1 = copy.deepcopy(self.policy.mlp_extractor.shared_net[0].weight.grad.detach().numpy())
            #inference_g_1 = copy.deepcopy(self.inference_net.fc_mean.weight.grad.detach().numpy())
            #embedding_g_1 = copy.deepcopy(self.embedding_net.fc_mean.weight.grad.detach().numpy())

            #import math
            #if math.isnan(policy_g_1[0, 0]) or math.isnan(inference_g_1[0, 0]) or math.isnan(embedding_g_1[0, 0]):
            #    print('policy', policy_g_1)
            #    print('inference', inference_g_1)
            #    print('embedding', embedding_g_1)
            #    print('')

            # policy parameter optimization
            self.policy.optimizer.zero_grad()
            self.inference_optimizer.zero_grad()
            self.embedding_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            policy_g_2 = copy.deepcopy(self.policy.mlp_extractor.shared_net[0].weight.grad.detach().numpy())
            inference_g_2 = copy.deepcopy(self.inference_net.fc_mean.weight.grad.detach().numpy())
            embedding_g_2 = copy.deepcopy(self.embedding_net.fc_mean.weight.grad.detach().numpy())

            import math
            if math.isnan(policy_g_2[0, 0]) or math.isnan(inference_g_2[0, 0]) or math.isnan(embedding_g_2[0, 0]):
                print('nan occured 2')
            else:
                self.policy.optimizer.step()

            # inference parameter optimization
            self.policy.optimizer.zero_grad()
            self.inference_optimizer.zero_grad()
            self.embedding_optimizer.zero_grad()
            inference_loss.backward()
            th.nn.utils.clip_grad_norm_(self.inference_net.parameters(), self.max_grad_norm)

            policy_g_3 = copy.deepcopy(self.policy.mlp_extractor.shared_net[0].weight.grad.detach().numpy())
            inference_g_3 = copy.deepcopy(self.inference_net.fc_mean.weight.grad.detach().numpy())
            embedding_g_3 = copy.deepcopy(self.embedding_net.fc_mean.weight.grad.detach().numpy())

            import math
            if math.isnan(policy_g_3[0, 0]) or math.isnan(inference_g_3[0, 0]) or math.isnan(embedding_g_3[0, 0]):
                print('nan occured 3')
            else:
                self.inference_optimizer.step()
            
            error_zs_each_cpu = rollout_data.error_zs
            error_zs = error_zs_each_cpu.mean(dim=[0, 1])

        #explained_var = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())

        self._n_updates += 1
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        #logger.record("train/explained_variance", explained_var)
        #logger.record("train/loss", loss.item())
        logger.record("train/policy_loss", policy_loss.item())
        logger.record("train/inference_loss", inference_loss.item())
        logger.record("train/embedding_loss", embedding_loss.item())
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        
        self.policy_loss_history.append(policy_loss.item())
        self.inference_loss_history.append(inference_loss.item())
        self.embedding_loss_history.append(embedding_loss.item())
        self.reward_history.append(reward_sum.mean().item())
        self.R_history.append(R_for_h.mean().item())
        self.error_zs_history.append(error_zs.detach().numpy().tolist())

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "A2C",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True
    ) -> "A2C":

        return super(A2C, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps
        )
    
    def reset_task_ids(self, task_id_list):
        for e, task_id in zip(self.env.envs, task_id_list):
            e.env.reset(task_id)

        self.env_task_id_list = [e.env.task_id for e in self.env.envs]
        self.task_id_list = task_id_list
        

    #def save(self, file_name):
    #    with open(file_name, 'wb') as f:
    #        pickle.dump(self, f)
    
    #@classmethod
    #def load(cls, file_name):
    #    with open(file_name, 'rb') as f:
    #        model = pickle.load(f)
    #    
    #    return model

