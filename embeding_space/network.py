import torch as th
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal

from stable_baselines3.common.distributions import DiagGaussianDistribution

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


class InferenceNet(nn.Module):
    def __init__(self, observation_space, action_space, embeding_dim, device='cpu'):
        super(InferenceNet, self).__init__()

        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        self.input_dim = self.observation_space + self.action_space
        self.output_dim = embeding_dim

        self.fc1 = nn.Linear(self.input_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc_mean = nn.Linear(200, self.output_dim)
        self.fc_var = nn.Linear(200, self.output_dim)

    def encoder(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_var(x)

        return mean, log_std
    
    def reparameterize(self, mean, log_var):
        std = th.exp(0.5*log_var)
        eps = th.randn_like(std)
        return mean + eps*std

    def log_prob(self, distribution, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions: (th.Tensor)
        :return: (th.Tensor)
        """
        log_prob = distribution.log_prob(actions)
        return sum_independent_dims(log_prob)
    
    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        mean, log_std = self.encoder(x)
        std = th.ones_like(mean) * log_std.exp()
        distribution = Normal(mean, std)
        z = distribution.rsample()
        log_prob = self.log_prob(distribution, z)

        return z, log_prob

    def predict(self, x):
        x = x.reshape(-1, self.input_dim)
        mean, log_std = self.encoder(x)
        std = th.ones_like(mean) * log_std.exp()
        distribution = Normal(mean, std)
        z = mean
        log_prob = self.log_prob(distribution, z)

        return z, log_prob


class EmbedingNet(nn.Module):
    def __init__(self, task_id_dim, embeding_dim, device='cpu'):
        super(EmbedingNet, self).__init__()
        self.device = device
        self.task_id_dim = task_id_dim
        self.input_dim = self.task_id_dim
        self.output_dim = embeding_dim

        self.fc1 = nn.Linear(self.input_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc_mean = nn.Linear(200, self.output_dim)
        self.fc_std = nn.Linear(200, self.output_dim)

    def encoder(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_std(x)

        return mean, log_std

    def entropy(self, distribution) -> th.Tensor:
        return sum_independent_dims(distribution.entropy())
    
    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        mean, log_std = self.encoder(x)
        std = th.ones_like(mean) * log_std.exp()
        distribution = Normal(mean, std)
        z = distribution.rsample()
        entropy = self.entropy(distribution)

        return z, entropy

    def predict(self, x):
        x = x.reshape(-1, self.input_dim)
        mean, log_std = self.encoder(x)
        std = th.ones_like(mean) * log_std.exp()
        distribution = Normal(mean, std)
        z = mean
        entropy = self.entropy(distribution)

        return z, entropy



class PolicyNet(nn.Module):
    def __init__(self, observation_space, embeding_dim, action_space, device='cpu'):
        super(PolicyNet, self).__init__()
        self.device = device
        self.observation_space = observation_space
        self.input_dim = self.observation_space + embeding_dim
        self.output_dim = action_space[0]

        self.fc1 = nn.Linear(self.input_dim, 200)
        self.fc2 = nn.Linear(200, 100)

        self.fc_mean = nn.Linear(100, self.output_dim)
        self.fc_var = nn.Linear(100, self.output_dim)
        self.v = nn.Linear(100, 1)
        self.action_dist = DiagGaussianDistribution(self.action_dim)
        #self.v_act = torch.nn.Hardtanh()

    def encoder(self, x):
        mean = self.fc_mean(x)
        log_var = self.fc_var(x)

        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        std = th.exp(0.5*log_var)
        eps = th.randn_like(std)
        return mean + eps*std
    
    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        mean, log_var = self.encoder(x)
        prob = self.reparameterize(mean, log_var)

        value = self.v(x)

        return prob, value


if __name__=="__main__":
    
    observation_space = (3, )
    action_space = (3, )
    embeding_dim = 2
    task_id_dim = 2

    #policy = PolicyNet(observation_space[0], embeding_dim, action_space[0])
    inference_net = InferenceNet(observation_space[0], action_space[0], embeding_dim)
    embeding_net = EmbedingNet(task_id_dim, embeding_dim)

    task_id = [0, 1]
    z_e = embeding_net(th.tensor(task_id).float())

    obs = th.tensor([[0, 1, 2], [7, 8, 9]]).float()
    action =  th.tensor([[3, 4, 5], [10, 11, 12]]).float()
    in_data = th.cat([obs, action], axis=1)
    z_i = inference_net(in_data)

    #in_data = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).float()
    #prob, value = policy(in_data)
    print('')
