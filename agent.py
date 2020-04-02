import random
import numpy as np
import os
from .train import transform_state
import torch
DEVICE = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

class Agent:
    def __init__(self, path="agent.pkl"):
        self.model = torch.load(__file__[:-8] + path)
        
    def act(self, state):
        state = torch.tensor(transform_state(state)).to(DEVICE).float()
        return int(torch.argmax(self.model(state)))

    def reset(self):
        pass

