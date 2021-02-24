# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Beta
# from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# from network import Net
# import numpy as np


# class Agent():
#     """
#     Agent for training
#     """
#     max_grad_norm = 0.5
#     clip_param = 0.1  # epsilon in clipped loss
#     ppo_epoch = 10
#     buffer_capacity, batch_size = 2000, 128

#     def __init__(self, args):

#         use_cuda = torch.cuda.is_available()
#         self.device = torch.device("cuda" if use_cuda else "cpu")
#         torch.manual_seed(args.seed)
#         if use_cuda:
#             torch.cuda.manual_seed(args.seed)

#         transition = np.dtype(
#             [('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
#              ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96))])

#         self.training_step = 0
#         self.net = Net().double().to(self.device)
#         self.buffer = np.empty(self.buffer_capacity, dtype=transition)
#         self.counter = 0
#         self.gamma = args.gamma

#         self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

#     def select_action(self, state):
#         state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
#         with torch.no_grad():
#             alpha, beta = self.net(state)[0]
#         dist = Beta(alpha, beta)
#         action = dist.sample()
#         a_logp = dist.log_prob(action).sum(dim=1)

#         action = action.squeeze().cpu().numpy()
#         a_logp = a_logp.item()
#         return action, a_logp

#     def save_param(self):
#         torch.save(self.net.state_dict(), 'param/ppo_net_params.pkl')

#     def store(self, transition):
#         self.buffer[self.counter] = transition
#         self.counter += 1
#         if self.counter == self.buffer_capacity:
#             self.counter = 0
#             return True
#         else:
#             return False

#     def update(self):
#         self.training_step += 1

#         s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
#         a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
#         r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
#         s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)

#         old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

#         with torch.no_grad():
#             target_v = r + self.gamma * self.net(s_)[1]
#             adv = target_v - self.net(s)[1]
#             # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

#         for _ in range(self.ppo_epoch):
#             for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
#                 alpha, beta = self.net(s[index])[0]
#                 dist = Beta(alpha, beta)
#                 a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
#                 ratio = torch.exp(a_logp - old_a_logp[index])

#                 surr1 = ratio * adv[index]
#                 surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
#                 action_loss = -torch.min(surr1, surr2).mean()
#                 value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
#                 loss = action_loss + 2. * value_loss

#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
#                 self.optimizer.step()

#     def load_param(self):
#         self.net.load_state_dict(torch.load('param/ppo_net_params.pkl'))
import argparse
import os
from collections import namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.distributions import Normal
from vae import VanillaVAE
'''
Implementation of soft actor critic
Original paper: https://arxiv.org/abs/1801.01290
Not the author's implementation !
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()

parser.add_argument("--env_name", default="carla-v0")  # OpenAI gym environment name
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--gradient_steps', default=1, type=int)

parser.add_argument('--learning_rate', default=3e-3, type=int)
parser.add_argument('--gamma', default=0.99, type=int)  # discount gamma
parser.add_argument('--capacity', default=16, type=int)  # replay buffer size
parser.add_argument('--iteration', default=100000, type=int)  # num of  games
parser.add_argument('--batch_size', default=16, type=int)  # mini batch size
parser.add_argument('--seed', default=1, type=int)

# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--num_hidden_units_per_layer', default=256, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=100, type=int)  #
parser.add_argument('--load', default=False, type=bool)  # load model
args = parser.parse_args()

Transition = namedtuple('Transition', ['original_s', 's', 'a', 'r', 's_', 'd'])


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


# env = NormalizedActions(gym.make(args.env_name))

# # Set seeds
# env.seed(args.seed)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)

# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# max_action = float(env.action_space.high[0])
# min_Val = torch.tensor(1e-7).float()
# Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_ub, action_lb, min_log_std=-20, max_log_std=2):
        super(Actor, self).__init__()
        self.conv2d1 = nn.Conv2d(1, 32, 4)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2d2 = nn.Conv2d(32, 64, 4)
        self.pool2 = nn.MaxPool2d(2)
        self.conv2d3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2)
        self.conv2d4 = nn.Conv2d(128, state_dim, 3)
        self.pool4 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(4*state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        self.action_ub = action_ub
        self.action_lb = action_lb
        self.state_dim = state_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        # if torch.isnan(x).any():
        #     print("Actor_x")
        # x = torch.unsqueeze(x, 1)
        # print(x.shape)
        # x = self.pad1(x)
        x = self.conv2d1(x)
        x = self.pool1(x)
        x = self.conv2d2(x)
        x = self.pool2(x)
        x = self.conv2d3(x)
        x = self.pool3(x)
        x = self.conv2d4(x)
        x = self.pool4(x)
        x = x.flatten(1)
        encoded_state = x

        x = x.reshape(-1, 4*self.state_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        # log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)

        # s = torch.nn.Softplus()
        # sigma = s(log_std_head) + 1e-5
        sigma = torch.exp(log_std_head)
        dist = Normal(mu, sigma)
        z = dist.sample()
        return z, mu, log_std_head, dist.log_prob(z).sum(), encoded_state
        # return mu, log_std_head


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(4*state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        # self.fc3 = nn.Linear(256, action_dim)
        self.state_dim = state_dim

    def forward(self, x):
        # if torch.isnan(x).any():
        #     print("Critic_x")
        x = x.reshape(-1, 4*self.state_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim*4 + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        # self.fc3 = nn.Linear(256, action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, s, a):
        # if torch.isnan(s).any():
        #     print("Q_s")
        # elif torch.isnan(a).any():
        #     print('Q_a')
        s = s.reshape(-1, self.state_dim*4)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Predict(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Predict, self).__init__()
        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim*4, state_dim*4)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim*4)
        x = self.fc1(s)
        return x


# class Cnn(nn.Module):
#     def __init__(self, state_dim):
#         super(Cnn, self).__init__()
#         # self.pad1 = nn.ZeroPad2d(2)
#         self.conv2d1 = nn.Conv2d(1,32,4)
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2d2 = nn.Conv2d(32,64,4)
#         self.pool2 = nn.MaxPool2d(2)
#         self.conv2d3 = nn.Conv2d(64,128,3)
#         self.pool3 = nn.MaxPool2d(2)
#         self.conv2d4 = nn.Conv2d(128,state_dim,3)
#         self.pool4 = nn.MaxPool2d(2)
#         # self.fc1 = nn.Linear(13, 13)
#         # self.fc2 = nn.Linear(13, 8)
#         # self.fc3 = nn.Linear(8, 4)
#         # self.fc4 = nn.Linear(4 ,1)
#
#     def forward(self, x):
#         x = torch.unsqueeze(x,1)
#         # print(x.shape)
#         # x = self.pad1(x)
#         x = self.conv2d1(x)
#         x = self.pool1(x)
#         x = self.conv2d2(x)
#         x = self.pool2(x)
#         x = self.conv2d3(x)
#         x = self.pool3(x)
#         x = self.conv2d4(x)
#         x = self.pool4(x)
#         x = x.flatten(1)
#         # print(x.shape)
#         return x


class SAC():
    def __init__(self, state_dim, action_dim, action_ub, action_lb, min_val):
        super(SAC, self).__init__()
        self.min_val = min_val
        self.action_ub = action_ub
        self.action_lb = action_lb

        # self.cnn_net = Cnn(state_dim)
        # self.cnn_optimizer = optim.Adam(self.cnn_net.parameters(), lr=1e-3)

        self.policy_net = Actor(state_dim, action_dim, action_ub, action_lb).to(device)
        self.value_net = Critic(state_dim, action_dim).to(device)

        self.Q_net1 = Q(state_dim, action_dim).to(device)
        self.Q_net2 = Q(state_dim, action_dim).to(device)

        self.Target_value_net = Critic(state_dim, action_dim).to(device)
        # self.predict_net = Predict(state_dim, action_dim).to(device)

        self.replay_buffer = [Transition] * args.capacity
        self.vae_replay_buffer = [Transition] * 256
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.learning_rate)

        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=args.learning_rate)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=args.learning_rate)

        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1
        self.writer = SummaryWriter('./exp-SAC')

        self.value_criterion = nn.MSELoss()

        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./SAC_model/', exist_ok=True)

        self.vae = vae = VanillaVAE(1, 256).to(device).float()
        self.vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    def predict_next_state(self, encoded_state):
        state_ = self.predict_net(encoded_state)
        reward = self.value_net(state_)
        return state_, reward

    def update_vae(self, state):
        self.vae_optimizer.zero_grad()
        reconstructed_state, _, mu, log_var = self.vae(state)
        loss = self.vae.loss_function(reconstructed_state, state, mu, log_var)['loss']
        loss.backward()
        self.vae_optimizer.step()

    # def update_cnn(self, state):
    #     self.cnn_optimizer.zero_grad()

    def select_action(self, state):
        #self.update_vae(state)
        state = torch.FloatTensor(state).to(device).unsqueeze(1)
        # sampled_latent_var = self.cnn_net(state)
        with torch.no_grad():
            # sampled_latent_var = self.encode_state(state)
            # mu, log_sigma = self.policy_net(sampled_latent_var)
            z, _, _, _, sampled_latent_var = self.policy_net(state)
        # sigma = torch.exp(log_sigma)
        # dist = Normal(mu, sigma)
        # z = dist.sample()

        # z = torch.tanh(z)
        if torch.FloatTensor(1, 1).uniform_() > 0.9:
            action_0 = torch.FloatTensor(1, 1).uniform_(self.action_lb[0], self.action_ub[0])
            action_1 = torch.FloatTensor(1, 1).uniform_(self.action_lb[1], self.action_ub[1])
        else:
            # action_0 = (torch.tanh(z[:, 0]) * self.action_ub[0]).unsqueeze(1)
            # action_1 = (torch.tanh(z[:, 1]) * self.action_ub[1]).unsqueeze(1)
            # to do
            action_0 = torch.clamp(z[:, 0], self.action_lb[0], self.action_ub[0]).unsqueeze(1)
            action_1 = torch.clamp(z[:, 1], self.action_lb[1], self.action_ub[1]).unsqueeze(1)
        action = torch.cat((action_0, action_1), dim=1).T.detach().cpu().numpy()
        #self.update_vae(state)
        return action, sampled_latent_var  # return a scalar, float32

    # def encode_state(self, state):
    #     with torch.no_grad():
    #         alpha, beta = self.vae.encode(state)
    #         sampled_latent_var = self.vae.reparameterize(alpha, beta)
    #     return sampled_latent_var
    def encode_state(self, state):
        with torch.no_grad():
            _, _, _, _, sampled_latent_var = self.policy_net(state)
        return sampled_latent_var

    def decode_state(self, sampled_latent_var):
        return self.vae.decode_single_input(sampled_latent_var)

    def store(self, original_s, s, a, r, s_, d):
        index = self.num_transition % args.capacity
        transition = Transition(original_s, s, a, r, s_, d)
        self.replay_buffer[index] = transition

        position = self.num_transition % 256
        self.vae_replay_buffer[position] = transition
        self.num_transition += 1

    def get_action_log_prob(self, state):
        state = state.view(64, 64, 64).cpu()
        state = torch.FloatTensor(state).to(device).unsqueeze(1)
        with torch.no_grad():
            z, batch_mu, batch_log_sigma, log_prob, encoded_state = self.policy_net(state)
        # batch_sigma = torch.exp(batch_log_sigma)
        # dist = Normal(batch_mu, batch_sigma)
        # z = dist.sample()
        # action_0 = (torch.tanh(z[:, 0]) * self.action_ub[0]).unsqueeze(1)
        # action_1 = (torch.tanh(z[:, 1]) * self.action_ub[1]).unsqueeze(1)
        action_0 = torch.clamp(z[:, 0], self.action_lb[0], self.action_ub[0]).unsqueeze(1)
        action_1 = torch.clamp(z[:, 1], self.action_lb[1], self.action_ub[1]).unsqueeze(1)
        action = torch.cat((action_0, action_1), dim=1).T
        # log_prob = dist.log_prob(z)
        # action = torch.tanh(action)
        # bound = torch.tensor([[3], [0.3]])

        # log_prob = dist.log_prob(z) - torch.log(bound - action.pow(2) + 1e-7).T
        return action, log_prob, z, batch_mu, batch_log_sigma, encoded_state

    def vae_update(self):
        if self.num_training % 256 == 0:
            print("Training ... {} ".format(self.num_training))
        original_s = torch.tensor([t.original_s for t in self.vae_replay_buffer]).float().to(device)
        original_s = original_s.permute(2, 3, 0, 1)
        original_s = original_s.view(64, 64, -1)
        original_s = original_s.permute(2, 0, 1)
        original_s = torch.unsqueeze(original_s, 1)
        # original_s = torch.from_numpy(original_s)
        self.update_vae(original_s)

    def update(self):
        if self.num_training % 16 == 0:
            print("Training ... {} ".format(self.num_training))
        original_s = torch.tensor([t.original_s for t in self.replay_buffer]).float().to(device)
        s = torch.tensor([t.s.cpu().numpy() for t in self.replay_buffer]).float().to(device)
        a = torch.tensor([t.a for t in self.replay_buffer]).float().to(device)
        r = torch.tensor([t.r for t in self.replay_buffer]).float().to(device)
        s_ = torch.tensor([t.s_.cpu().numpy() for t in self.replay_buffer]).float().to(device)
        d = torch.tensor([t.d for t in self.replay_buffer]).float().to(device)

        # original_s = original_s.permute(2, 3, 0, 1)
        # original_s = original_s.view(64, 64, -1)
        # original_s = original_s.permute(2, 0, 1)
        # original_s = torch.unsqueeze(original_s, 1)

        # original_s = torch.from_numpy(original_s)
        # self.update_vae(original_s)
        for _ in range(args.gradient_steps):
            # for index in BatchSampler(SubsetRandomSampler(range(args.capacity)), args.batch_size, False):
            index = np.random.choice(range(args.capacity), args.batch_size, replace=False)
            bn_original_s = original_s[index]
            bn_s = s[index]
            bn_a = a[index].reshape(-1, 1)
            bn_r = r[index].reshape(-1, 1)
            bn_s_ = s_[index]
            bn_d = d[index].reshape(-1, 1)

            target_value = self.Target_value_net(bn_s_)
            next_q_value = bn_r + (1 - bn_d) * args.gamma * target_value

            excepted_value = self.value_net(bn_s)
            excepted_Q1 = self.Q_net1(bn_s, bn_a)
            excepted_Q2 = self.Q_net2(bn_s, bn_a)

            sample_action, log_prob, z, batch_mu, batch_log_sigma, bn_encoded_s = self.get_action_log_prob(bn_original_s)
            excepted_new_Q = torch.min(self.Q_net1(bn_encoded_s, sample_action), self.Q_net2(bn_encoded_s, sample_action))
            next_value = excepted_new_Q - log_prob

            # !!!Note that the actions are sampled according to the current policy,
            # instead of replay buffer. (From original paper)

            V_loss = self.value_criterion(excepted_value, next_value.detach())  # J_V
            V_loss = V_loss.mean()


            # Single Q_net this is different from original paper!!!
            Q1_loss = self.Q1_criterion(excepted_Q1, next_q_value.detach()).mean()  # J_Q
            Q2_loss = self.Q2_criterion(excepted_Q2, next_q_value.detach()).mean()

            pi_loss = (log_prob - excepted_new_Q).mean()
            # log_policy_target = excepted_new_Q - excepted_value
            # pi_loss = log_prob * (log_prob - log_policy_target)
            # pi_loss = pi_loss.mean()
            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            self.writer.add_scalar('Loss/V_loss', V_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q1_loss', Q1_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q2_loss', Q2_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/pi_loss', pi_loss, global_step=self.num_training)
            # mini batch gradient descent
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.Q1_optimizer.zero_grad()
            Q1_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Q1_optimizer.step()

            self.Q2_optimizer.zero_grad()
            Q2_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            self.Q2_optimizer.step()

            # log_policy_target = excepted_new_Q - excepted_value
            # pi_loss = log_prob * (log_prob - log_policy_target)
            # pi_loss = pi_loss.mean()
            # self.policy_optimizer.zero_grad()
            # pi_loss.backward(retain_graph=True)
            # nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            # self.policy_optimizer.step()

            # soft update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - args.tau) + param * args.tau)

            self.num_training += 1

    def save(self):
        torch.save(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.save(self.value_net.state_dict(), './SAC_model/value_net.pth')
        torch.save(self.Q_net1.state_dict(), './SAC_model/Q_net1.pth')
        torch.save(self.Q_net2.state_dict(), './SAC_model/Q_net2.pth')
        torch.save(self.vae.state_dict(), './SAC_model/vae.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.policy_net.load_state_dict(torch.load('./SAC_model/policy_net.pth'))
        self.value_net.load_state_dict(torch.load('./SAC_model/value_net.pth'))
        self.Q_net1.load_state_dict(torch.load('./SAC_model/Q_net1.pth'))
        self.Q_net2.load_state_dict(torch.load('./SAC_model/Q_net2.pth'))
        self.vae.load_state_dict(torch.load('./SAC_model/vae.pth'))
        print("model has been loaded")
