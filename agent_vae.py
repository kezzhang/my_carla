import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from policy_net_vae_input import VAEInputNet
from gae import VanillaVAE
from b_vae import BetaVAE

class VAEAgent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 1000, 128

    def __init__(self, args, z_size):

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)

        transition = np.dtype(
            [('s', np.float64, (4, z_size)), ('a', np.float64, (3,)), ('a_logp', np.float64),
             ('r', np.float64), ('s_', np.float64, (4,z_size))])

        self.training_step = 0
        self.net = VAEInputNet(z_size, args.img_stack).double().to(self.device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        self.gamma = args.gamma
        self.z_size = z_size
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.vae = vae = BetaVAE(1, z_size).double().to(self.device)
        self.vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)
        self.vae.load_state_dict(torch.load('param/b2_vae.pkl'))


    def update_vae(self, state_tensor):
        self.vae_optimizer.zero_grad()
        reconstructed_state,_, mu, log_var = self.vae(state_tensor)
        loss = self.vae.loss_function(reconstructed_state, state_tensor, mu, log_var)['loss']
        loss.backward()
        self.vae_optimizer.step()

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(1)
        with torch.no_grad():
            mu,log_var= self.vae.encode(state)
            sampled_latent_var = self.vae.reparameterize(mu,log_var)
            alpha, beta = self.net(sampled_latent_var)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        #self.update_vae(state)
        return action, a_logp

    def save_param(self):
        torch.save(self.net.state_dict(), 'param/ppo_net_params_b2vae.pkl')
        torch.save(self.vae.state_dict(), 'param/ppo_b2vae.pkl')

    def store_transition(self, state, action, a_logp, reward, state_):
        with torch.no_grad():
            state = torch.from_numpy(state).double().to(self.device).unsqueeze(1)
            state_ = torch.from_numpy(state_).double().to(self.device).unsqueeze(1)

            mu,log_var= self.vae.encode(state)
            latent_var = self.vae.reparameterize(mu,log_var).cpu().numpy()
            mu_, log_var_ = self.vae.encode(state_)
            latent_var_ = self.vae.reparameterize(mu_, log_var_).cpu().numpy()
        self.buffer[self.counter] = (latent_var,action, a_logp, reward, latent_var_)
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = r + self.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def load_param(self):
        self.vae.load_state_dict(torch.load('param/vae_only_vae.pkl'))
        self.net.load_state_dict(torch.load('param/ppo_only_vae.pkl'))
