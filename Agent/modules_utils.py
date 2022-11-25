import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ConvNN(nn.Module):
    def __init__(self, n_channels):
        super(ConvNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv(x)
    
class FCNN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(FCNN, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU())
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.fc(x)
        return self.policy(x), self.value(x)
    
    
class RolloutEncoder(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(RolloutEncoder, self).__init__()
        
        self.conv = ConvNN(input_shape[0])
        conv_out_size = self._get_conv_out(input_shape)
        self.rnn = nn.LSTM(input_size = conv_out_size + 1,
                           hidden_size = hidden_size,
                           batch_first = False)
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, states, rewards):
        n_time, n_batch = states.size()[0], states.size()[1]
        n_items =  n_time*n_batch
        states_f = states.view(n_items, *states.size()[2:])
        conv_out = self.conv(states_f).view(n_time, n_batch, -1)
        rnn_in = torch.cat((conv_out, rewards), dim=2)
        _, (rnn_hid, _) = self.rnn(rnn_in)
        return rnn_hid.view(-1)
        