import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def broadcast_actions(actions, size):
    act_broadcasted = torch.FloatTensor(*size).to(actions.device)
    act_broadcasted.zero_()
    act_broadcasted[range(size[0]), actions] = 1.0
    return act_broadcasted

class EnvModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(EnvModel, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        n_plates = input_shape[0] + n_actions
        self.enc1 = nn.Sequential(
            nn.Conv2d(n_plates, 64, kernel_size=4, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dec_img = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=4, padding=0)
        self.dec_rew = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        reward_dec_size = self._get_reward_dec_size((n_plates, ) + input_shape[1:])
        self.dec_rew_fc = nn.Sequential(
            nn.Linear(reward_dec_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def _get_reward_dec_size(self, shape):
        o = self.enc1(torch.zeros(1, *shape))
        o = self.dec_rew(o)
        return int(np.prod(o.size()))
    
    def forward(self, imgs, actions):
        batch_size = actions.size()[0]
        emb_size = (batch_size, self.n_actions, ) + self.input_shape[1:]
        x = torch.cat((imgs, broadcast_actions(actions, emb_size)), dim=1)
        z1 = self.enc1(x)
        z2 = z1 + self.enc2(z1)
        img_out = self.dec_img(z2)
        rew_out = self.dec_rew_fc(self.dec_rew(z2).view(batch_size, -1))
        return img_out, rew_out