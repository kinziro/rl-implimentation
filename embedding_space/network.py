import torch as th
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from functools import partial
import gym

from distributions import ReproduceNormal, ReproduceDiagGaussianDistribution, make_proba_distribution
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor, NatureCNN, create_mlp
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, preprocess_obs
from stable_baselines3.common.policies import ActorCriticPolicy, create_sde_features_extractor

def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: (th.Tensor) shape: (n_batch, n_actions) or (n_batch,)
    :return: (th.Tensor) shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

class BaseDistributionNet(nn.Module):
    def __init__(self, input_dim, output_dim, net_arch, normalization_coeff, device='cpu'):
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_arch = net_arch
        self.normalization_coeff = normalization_coeff
        if self.normalization_coeff is not None:
            self.normalization_coeff = th.tensor(normalization_coeff, requires_grad=False).reshape(1, -1)

        self.fc1 = nn.Linear(self.input_dim, net_arch[0])
        self.fc2 = nn.Linear(net_arch[0], net_arch[1])
        self.fc_mean = nn.Linear(net_arch[1], self.output_dim)
        self.fc_std = nn.Linear(net_arch[1], self.output_dim)

        self.sample_eps = None

    def encoder(self, x):
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = th.tanh(self.fc1(x))
        x = th.tanh(self.fc2(x))
        mean = th.tanh(self.fc_mean(x))
        #mean = self.fc_mean(x)
        log_std = self.fc_std(x)

        return mean, log_std
    
    def log_prob(self, distribution, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions: (th.Tensor)
        :return: (th.Tensor)
        """
        log_prob = distribution.log_prob(actions)
        return sum_independent_dims(log_prob)
    
    def base_forward(self, x, sampling_rule='sampling'):
        re_x = x.reshape(-1, self.input_dim)
        if self.normalization_coeff is not None:
            norm_x = re_x / self.normalization_coeff
        else:
            norm_x = re_x
        mean, log_std = self.encoder(norm_x)
        std = th.ones_like(mean) * log_std.exp()
        distribution = ReproduceNormal(mean, std)
        if sampling_rule=='deterministic':
            z = mean
        elif sampling_rule=='reproduce':
            z = distribution.reproduce_sample(self.sample_eps)
        else:
            z, self.sample_eps = distribution.rsample2()
        
        cut_z = th.tensor(z.detach().numpy())
        log_prob = self.log_prob(distribution, cut_z)

        return z, log_prob, distribution
    
    def forward(self, x, reproduce=False):
        sampling_rule = 'reproduce' if reproduce else 'sampling'
        z, log_prob, _ = self.base_forward(x, sampling_rule)

        return z, log_prob

    def predict(self, x):
        z, log_prob, _ = self.base_forward(x, 'deterministic')

        return z, log_prob

    def get_mean_std(self, x):
        re_x = x.reshape(-1, self.input_dim)
        mean, log_std = self.encoder(re_x)
        std = th.ones_like(mean) * log_std.exp()

        return mean, std

    def evaluate_actions(self, x: th.Tensor, z: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: (th.Tensor)
        :param actions: (th.Tensor)
        :return: (th.Tensor, th.Tensor, th.Tensor) estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        z = th.tensor(z.detach().numpy())

        re_x = x.reshape(-1, self.input_dim)
        if self.normalization_coeff is not None:
            norm_x = re_x / self.normalization_coeff
        else:
            norm_x = re_x
        mean, log_std = self.encoder(norm_x)
        std = th.ones_like(mean) * log_std.exp()
        distribution = ReproduceNormal(mean, std)
        log_prob = distribution.log_prob(z)

        return log_prob, distribution.entropy()

class InferenceNet2(BaseDistributionNet):
    def __init__(self, observation_space, action_space, embedding_dim, n_obs_history=1, net_arch=[100, 100], normalization_coeff=None, device='cpu'):

        self.observation_space = observation_space
        self.action_space = action_space
        self.n_obs_history = n_obs_history
        input_dim = (self.observation_space * self.n_obs_history) + (self.action_space * self.n_obs_history)
        output_dim = embedding_dim
        if normalization_coeff is not None:
            normalization_coeff = normalization_coeff[:observation_space] * n_obs_history + normalization_coeff[observation_space:]

        super().__init__(input_dim=input_dim, output_dim=output_dim, net_arch=net_arch, normalization_coeff=normalization_coeff, device=device)



class InferenceNet(BaseDistributionNet):
    def __init__(self, observation_space, action_space, embedding_dim, n_obs_history=1, net_arch=[100, 100], normalization_coeff=None, device='cpu'):

        self.observation_space = observation_space
        self.action_space = action_space
        self.n_obs_history = n_obs_history
        input_dim = (self.observation_space * self.n_obs_history) + self.action_space
        output_dim = embedding_dim
        if normalization_coeff is not None:
            normalization_coeff = normalization_coeff[:observation_space] * n_obs_history + normalization_coeff[observation_space:]

        super().__init__(input_dim=input_dim, output_dim=output_dim, net_arch=net_arch, normalization_coeff=normalization_coeff, device=device)


class EmbeddingNet(BaseDistributionNet):
    def __init__(self, task_id_dim, embedding_dim, net_arch=[100, 100], normalization_coeff=None, device='cpu'):
        self.task_id_dim = task_id_dim
        input_dim = self.task_id_dim
        output_dim = embedding_dim

        super().__init__(input_dim=input_dim, output_dim=output_dim, net_arch=net_arch, normalization_coeff=normalization_coeff, device=device)
        
    def entropy(self, distribution) -> th.Tensor:
        return sum_independent_dims(distribution.entropy())

    def forward(self, x, reproduce=False):
        sampling_rule = 'reproduce' if reproduce else 'sampling'
        z, log_prob, distribution = self.base_forward(x, sampling_rule)
        entropy = self.entropy(distribution)

        return z, log_prob, entropy

    def predict(self, x):
        z, log_prob, distribution = self.base_forward(x, 'deterministic')
        entropy = self.entropy(distribution)

        return z, log_prob, entropy

class EmbeddingNet2(nn.Module):
    def __init__(self, task_id_dim, env_id_dim, task_embedding_dim, env_embedding_dim, net_arch=[100, 100], normalization_coeff=None, device='cpu'):
        super().__init__()
        self.task_id_dim = task_id_dim
        output_dim = task_embedding_dim + env_embedding_dim

        # 以下コンストラクタ
        self.device = device
        self.input_dim_t = task_id_dim
        self.input_dim_e = env_id_dim
        self.output_dim_t = task_embedding_dim
        self.output_dim_e = env_embedding_dim
        self.net_arch = net_arch
        self.normalization_coeff = normalization_coeff
        if self.normalization_coeff is not None:
            self.normalization_coeff = th.tensor(normalization_coeff, requires_grad=False).reshape(1, -1)

        self.fc1_t = nn.Linear(self.input_dim_t, net_arch[0])
        self.fc2_t = nn.Linear(net_arch[0], net_arch[1])
        self.fc_mean_t = nn.Linear(net_arch[1], self.output_dim_t)
        self.fc_std_t = nn.Linear(net_arch[1], self.output_dim_t)

        self.fc1_e = nn.Linear(self.input_dim_e, net_arch[0])
        self.fc2_e = nn.Linear(net_arch[0], net_arch[1])
        self.fc_mean_e = nn.Linear(net_arch[1], self.output_dim_e)
        self.fc_std_e = nn.Linear(net_arch[1], self.output_dim_e)

        self.sample_eps = None

    def encoder_t(self, x):
        x = th.tanh(self.fc1_t(x))
        x = th.tanh(self.fc2_t(x))
        mean = th.tanh(self.fc_mean_t(x))
        log_std = self.fc_std_t(x)

        return mean, log_std

    def encoder_e(self, x):
        x = th.tanh(self.fc1_e(x))
        x = th.tanh(self.fc2_e(x))
        mean = th.tanh(self.fc_mean_e(x))
        log_std = self.fc_std_e(x)

        return mean, log_std
 
    def log_prob(self, distribution, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions: (th.Tensor)
        :return: (th.Tensor)
        """
        log_prob = distribution.log_prob(actions)
        return sum_independent_dims(log_prob)
    
    def base_forward(self, t_id, e_id, sampling_rule='sampling'):
        re_t_id = t_id.reshape(-1, self.input_dim_t)
        re_e_id = e_id.reshape(-1, self.input_dim_e)
        if self.normalization_coeff is not None:
            norm_x_t = re_t_id / self.normalization_coeff
            norm_x_e = re_e_id / self.normalization_coeff
        else:
            norm_x_t = re_t_id
            norm_x_e = re_e_id
        mean_t, log_std_t = self.encoder_t(norm_x_t)
        mean_e, log_std_e = self.encoder_e(norm_x_e)
        mean = th.cat([mean_t, mean_e], dim=1)
        log_std = th.cat([log_std_t, log_std_e], dim=1)

        std = th.ones_like(mean) * log_std.exp()
        distribution = ReproduceNormal(mean, std)
        if sampling_rule=='deterministic':
            z = mean
        elif sampling_rule=='reproduce':
            z = distribution.reproduce_sample(self.sample_eps)
        else:
            z, self.sample_eps = distribution.rsample2()
        
        cut_z = th.tensor(z.detach().numpy())
        log_prob = self.log_prob(distribution, cut_z)

        return z, log_prob, distribution
    
    def forward(self, t_id, e_id, reproduce=False):
        sampling_rule = 'reproduce' if reproduce else 'sampling'
        z, log_prob, distribution = self.base_forward(t_id, e_id, sampling_rule)
        entropy = self.entropy(distribution)

        return z, log_prob, entropy

    def predict(self, t_id, e_id):
        z, log_prob, distribution = self.base_forward(t_id, e_id, 'deterministic')
        entropy = self.entropy(distribution)

        return z, log_prob, entropy

    def get_mean_std(self, t_id, e_id):
        re_x_t = t_id.reshape(-1, self.input_dim_t)
        re_x_e = e_id.reshape(-1, self.input_dim_e)
        mean_t, log_std_t = self.encoder_t(re_x_t)
        mean_e, log_std_e = self.encoder_e(re_x_e)

        mean = th.cat([mean_t, mean_e], dim=1)
        log_std = th.cat([log_std_t, log_std_e], dim=1)
        std = th.ones_like(mean) * log_std.exp()

        return mean, std

    def evaluate_actions(self, t_id: th.Tensor, e_id: th.Tensor, z: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: (th.Tensor)
        :param actions: (th.Tensor)
        :return: (th.Tensor, th.Tensor, th.Tensor) estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        z = th.tensor(z.detach().numpy())

        re_x_t = t_id.reshape(-1, self.input_dim_t)
        re_x_e = e_id.reshape(-1, self.input_dim_e)
        if self.normalization_coeff is not None:
            norm_x_t = re_t_id / self.normalization_coeff
            norm_x_e = re_e_id / self.normalization_coeff
        else:
            norm_x_t = re_t_id
            norm_x_e = re_e_id

        mean_t, log_std_t = self.encoder_t(re_x_t)
        mean_e, log_std_e = self.encoder_e(re_x_e)
        mean = th.cat([mean_t, mean_e], dim=1)
        log_std = th.cat([log_std_t, log_std_e], dim=1)

        std = th.ones_like(mean) * log_std.exp()
        distribution = ReproduceNormal(mean, std)
        log_prob = distribution.log_prob(z)

        return log_prob, distribution.entropy()
        
    def entropy(self, distribution) -> th.Tensor:
        return sum_independent_dims(distribution.entropy())


class PolicyNet(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        device: Union[th.device, str] = "auto",
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        normalization_coeff: list = None
    ):
        self.normalization_coeff = normalization_coeff
        if self.normalization_coeff is not None:
            self.normalization_coeff = th.tensor(self.normalization_coeff, requires_grad=False).reshape(1, -1)


        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            device,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]
            else:
                net_arch = []

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": sde_net_arch is not None,
            }

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)


    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: (Callable) Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device
        )

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate feature extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_dist, ReproduceDiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # feature_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        #self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

        self.sample_eps = None

    def base_forward(self, obs: th.Tensor, sampling_rule: str = 'sampling') -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: (th.Tensor) Observation
        :param deterministic: (bool) Whether to sample or use deterministic actions
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) action, value and log probability of the action
        """
        if self.normalization_coeff is not None:
            norm_obs = obs / self.normalization_coeff
        else:
            norm_obs = obs
        latent_pi, latent_vf, latent_sde = self._get_latent(norm_obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        if sampling_rule=='deterministic':
            actions = distribution.get_actions(deterministic=True)
        elif sampling_rule=='reproduce':
            actions = distribution.reproduce_sample(self.sample_eps)
        else:
            actions, self.sample_eps = distribution.get_actions2()

        cut_actions = th.tensor(actions.detach().numpy())
        log_prob = distribution.log_prob(cut_actions)
        return actions, values, log_prob, distribution.entropy()

    def forward(self, x, reproduce=False, deterministic=False):
        if deterministic:
            sampling_rule = 'deterministic'
        else:
            sampling_rule = 'reproduce' if reproduce else 'samping'
        actions, values, log_prob, entropy = self.base_forward(x, sampling_rule)

        return actions, values, log_prob, entropy

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: (th.Tensor)
        :param actions: (th.Tensor)
        :return: (th.Tensor, th.Tensor, th.Tensor) estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        actions = th.tensor(actions.detach().numpy())

        if self.normalization_coeff is not None:
            norm_obs = obs / self.normalization_coeff
        else:
            norm_obs = obs
        latent_pi, latent_vf, latent_sde = self._get_latent(norm_obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: (th.Tensor) Latent code for the actor
        :param latent_sde: (Optional[th.Tensor]) Latent code for the gSDE exploration function
        :return: (Distribution) Action distribution
        """
        mean_actions = th.tanh(self.action_net(latent_pi))

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")


class StateEmbeddingMappingNet(nn.Module):
    def __init__(self, input_dim, output_dim, net_arch, normalization_coeff=None, device='cpu'):
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_arch = net_arch
        self.normalization_coeff = normalization_coeff
        if self.normalization_coeff is not None:
            self.normalization_coeff = th.tensor(normalization_coeff, requires_grad=False).reshape(1, -1)

        self.fc1 = nn.Linear(self.input_dim, net_arch[0])
        self.fc2 = nn.Linear(net_arch[0], net_arch[1])
        self.fc3 = nn.Linear(net_arch[1], self.output_dim)

    def forward(self, x):
        x = th.tanh(self.fc1(x))
        x = th.tanh(self.fc2(x))
        z = th.tanh(self.fc_3(x))

        return z
