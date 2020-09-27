import torch.nn as nn


class VAEInputNet(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, z_size, img_stack):
        super(VAEInputNet, self).__init__()
        self.z_size = z_size
        self.img_stack = img_stack
        self.v = nn.Sequential(nn.Linear(z_size * img_stack, 200), nn.ReLU(),
                               nn.Linear(200, 100), nn.ReLU(),
                               nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(z_size * img_stack, 200), nn.ReLU(),
                                nn.Linear(200, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = x.view(-1, self.z_size * self.img_stack)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v
