import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Agent.utils import broadcast_actions


class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)
        
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        fc_out = self.fc(conv_out)
        return self.policy(fc_out), self.value(fc_out)
    
    

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