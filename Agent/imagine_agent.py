import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.utils as nn_utils
import numpy as np


from Agent.modules import ActorCritic, I2A
from Agent.utils import preprocess_states, unpack_batch, analyse_logits, rollout_init, buil_state_from_delta

class ImagineAgent:
    
    def __init__(self, obs_shape, n_actions, device, config, env_model, em_out_shape, load_path = None):
        
        self.i2a_net = I2A(obs_shape, n_actions, em_out_shape, config.hidden_size).to(device)
        self.i2a_optimizer = optim.RMSprop(self.i2a_net.parameters(), lr = config.lr_i2a, eps=1e-5)
        
        self.policy_net = ActorCritic(obs_shape, n_actions).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr_policy)
        
        self.env_model = env_model
        
        self.device = device
        self.beta = config.beta
        self.coef = config.coef
        self.clip_grad = config.clip_grad
        self.rollout_steps = config.rollout_steps
        self.n_actions = n_actions
        if load_path is not None:
            self.i2a_net.load_state_dict(torch.load(load_path+"i2a_weight.pt", map_location=torch.device(device)))
            self.policy_net.load_state_dict(torch.load(load_path+"policy_weight.pt", map_location=torch.device(device)))
        
        
    def predict(self, states):
         pred_states, pred_rewards = self.rollouts(states)
         return self.i2a_net(states, pred_states, pred_rewards)
        
    def __call__(self, states, only_values = False):
        states = preprocess_states(states, self.device)
        with torch.no_grad():
            logits, values = self.predict(states)
        if only_values: return values.cpu().numpy().squeeze()
        probs = F.softmax(logits, dim=1)
        actions = Categorical(logits=logits).sample()
        return values.cpu().numpy().squeeze(), actions.cpu().numpy().astype(np.int32), probs.cpu().numpy()
    
    def update_i2a(self, states, ref_values, advs, actions):
        logits, values = self.predict(states)
        probs, entropy = analyse_logits(logits, actions)
        loss_value = F.mse_loss(torch.squeeze(values), ref_values)
        loss_policy = -(advs*probs).mean()
        loss = loss_policy + self.coef*loss_value - self.beta*entropy
        loss.backward()
        nn_utils.clip_grad_norm_(self.i2a_net.parameters(), self.clip_grad)
        self.i2a_optimizer.step()
        
    def update_policy(self, states, probs):
         logits, _ = self.policy_net(states)
         loss = -F.log_softmax(logits, dim=1)*probs.view_as(logits)
         loss = loss.sum(dim=1).mean()
         loss.backward()
         self.policy_optimizer.step()
         
    def update(self, batch):
        self.i2a_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        states, ref_values, advs, probs, actions = unpack_batch(batch, self.device)
        self.update_i2a(states, ref_values, advs, actions)
        self.update_policy(states, probs)
        
    def save(self, path):
        torch.save(self.i2a_net.state_dict(), f"{path}i2a_weight.pt")
        torch.save(self.policy_net.state_dict(), f"{path}policy_weight.pt")
        
    @torch.no_grad()
    def rollouts(self, states):
        #expand the states such that we can rollout with every possible action
        states, actions = rollout_init(states, self.n_actions)
        pred_deltas_l, pred_rewards_l = [], []
        for idx in range(self.rollout_steps):
            pred_deltas, pred_rewards = self.env_model(states, actions)
            pred_deltas_l.append(pred_deltas.detach())
            pred_rewards_l.append(pred_rewards.detach())
            if idx == self.rollout_steps - 1: break
            states = buil_state_from_delta(states, pred_deltas)
            logits, _ = self.policy_net(states)
            actions = Categorical(logits=logits).sample()
        return torch.stack(pred_deltas_l), torch.stack(pred_rewards_l)