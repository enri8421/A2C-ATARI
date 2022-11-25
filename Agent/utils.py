import torch
import torch.nn.functional as F
import numpy as np


def preprocess_states(states, device):
    states_t = torch.FloatTensor(states) / 255
    return states_t.to(device)


def unpack_batch(batch, device):
    states_t = preprocess_states(batch.states, device)
    ref_values_t = torch.FloatTensor(batch.ref_values).to(device)
    advs_t = torch.FloatTensor(batch.advs).to(device)
    probs_t = torch.FloatTensor(batch.probs).to(device)
    actions_t = torch.LongTensor(batch.actions).to(device)
    return states_t, ref_values_t, advs_t, probs_t, actions_t

def unpack_transition(states, actions, rewards, dones, new_states, device):
    states_t = preprocess_states(states, device)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_t = torch.FloatTensor(rewards).to(device)
    dones_t = torch.BoolTensor(dones.reshape(-1,1,1)).to(device)
    delta_t = preprocess_states(new_states[:,1] - new_states[:,0], device)
    return states_t, actions_t, rewards_t, dones_t, delta_t

def analyse_logits(logits, actions):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    
    log_probs_actions = log_probs[range(len(actions)), actions]
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return log_probs_actions, entropy
    
def pred_state_loss(preds, deltas, dones):
    preds = torch.where(dones, torch.zeros_like(preds), preds)
    deltas = torch.where(dones, torch.zeros_like(deltas), deltas)
    loss = F.mse_loss(preds, deltas)
    return loss


def rollout_init(x, n):
    batch_size = x.size()[0]
    x_shape = x.size()[1:]
    
    if batch_size == 1:
        exp_x = x.expand(n, *x_shape)
    else:
        exp_x = x.unsqueeze(1)
        exp_x = exp_x.expand(batch_size, n, *x_shape)
        exp_x = exp_x.contiguous().view(-1, *x_shape)
        
    actions = np.tile(np.arange(0, n, dtype=np.int64), batch_size)
    actions_t = torch.LongTensor(actions).to(x.device)
    return exp_x, actions_t


def buil_state_from_delta(curr_state, delta):
    last_obs = curr_state[:, 1:2]
    next_obs = last_obs + delta
    next_state = torch.cat((last_obs, next_obs), dim=1)
    return next_state
    
        
    