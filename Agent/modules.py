import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Agent.modules_utils import ConvNN, FCNN, RolloutEncoder


class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        
        self.conv = ConvNN(input_shape[0])
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = FCNN(conv_out_size, n_actions)
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
    

class I2A(nn.Module):
    def __init__(self, input_shape, n_actions, em_out_shape, hidden_size):
        super(I2A, self).__init__()
        
        self.conv = ConvNN(input_shape[0])
        fc_input = self._get_conv_out(input_shape) + n_actions*hidden_size
        self.fc = FCNN(fc_input, n_actions)
        
        self.encoder = RolloutEncoder(em_out_shape, hidden_size)
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x, pred_states, pred_rewards):
        batch_size = x.size()[0]
        enc = self.encoder(pred_states, pred_rewards).view(batch_size, -1)
        conv_out = self.conv(x).view(batch_size, -1)
        fc_in = torch.cat((conv_out, enc), dim=1)
        return self.fc(fc_in)