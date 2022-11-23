import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.utils as nn_utils
import numpy as np


from Agent.module import ConvNN
from Agent.utils import preprocess_states, unpack_batch, analyse_logits

class Agent:
    
    def __init__(self, obs_shape, n_actions, device, beta, coef, lr, clip_grad):
        
        self.net = ConvNN(obs_shape, n_actions).to(device)
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr, eps=1e-5)
        self.device = device
        self.beta = beta
        self.coef = coef
        self.clip_grad = clip_grad
        
    def __call__(self, states, only_values = False):
        states = preprocess_states(states, self.device)
        with torch.no_grad():
            logits, values = self.net(states)
        if only_values: return values.cpu().numpy().squeeze()
        actions = Categorical(logits=logits).sample()
        return values.cpu().numpy().squeeze(), actions.cpu().numpy().astype(np.int32)
    
    
    def update(self, batch):
        self.optimizer.zero_grad()
        
        states, ref_values, advs, actions = unpack_batch(batch, self.device)
        logits, values = self.net(states)
        probs, entropy = analyse_logits(logits, actions)
        
        loss_value = F.mse_loss(torch.squeeze(values), ref_values)
        loss_policy = -(advs*probs).mean()
        loss = loss_policy + self.coef*loss_value - self.beta*entropy
        
        loss.backward()
        nn_utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)
        self.optimizer.step()
    
    def save(self, path):
        torch.save(self.net.state_dict(), f"{path}weight.pt")