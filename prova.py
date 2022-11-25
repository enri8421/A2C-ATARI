from Agent.base_agent import BaseAgent
from Agent.configuration import DefaultBaseAgent
import torch
from collections import OrderedDict

old = torch.load("weight.pt")

new = OrderedDict([("conv."+k if k[0] == 'c' else "fc."+k, v) for k, v in old.items()])

torch.save(new, "pweight.pt")