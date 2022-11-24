import torch
import torch.nn.functional as F


def preprocess_states(states, device):
    states_t = torch.FloatTensor(states) / 255
    return states_t.to(device)


def unpack_batch(batch, device):
    states_t = preprocess_states(batch.states, device)
    ref_values_t = torch.FloatTensor(batch.ref_values).to(device)
    advs_t = torch.FloatTensor(batch.advs).to(device)
    actions_t = torch.LongTensor(batch.actions).to(device)
    return states_t, ref_values_t, advs_t, actions_t

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
    
    
def broadcast_actions(actions, size):
    act_broadcasted = torch.FloatTensor(*size, device=actions.device)
    act_broadcasted.zero_()
    act_broadcasted[range(size[0]), actions] = 1.0
    return act_broadcasted


def pred_state_loss(preds, deltas, dones):
    preds = torch.where(dones, torch.zeros_like(preds), preds)
    deltas = torch.where(dones, torch.zeros_like(deltas), deltas)
    print(deltas.size(), preds.size())
    loss = F.mse_loss(preds, deltas)
    print(loss.size())
    return loss
    