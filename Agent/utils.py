import torch
import torch.nn.functional as F


def preprocess_states(states, device):
    states_t = torch.FloatTensor(states) / 256
    return states_t.to(device)


def unpack_batch(batch, device):
    states_t = preprocess_states(batch.states, device)
    ref_values_t = torch.FloatTensor(batch.ref_values).to(device)
    advs_t = torch.FloatTensor(batch.advs).to(device)
    actions_t = torch.LongTensor(batch.actions).to(device)
    return states_t, ref_values_t, advs_t, actions_t


def analyse_logits(logits, actions):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    
    log_probs_actions = log_probs[range(len(actions)), actions]
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return log_probs_actions, entropy
    