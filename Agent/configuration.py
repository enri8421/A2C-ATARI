


class DefaultBaseAgent:
    beta = 0.01 
    coef = 0.5 
    lr =  1e-4
    clip_grad = 0.5
    
    
class DefaultImagineAgent:
    beta = 0.01 
    coef = 0.5 
    lr_i2a =  1e-4
    lr_policy = 1e-4
    clip_grad = 0.5
    hidden_size = 256
    rollout_steps = 3