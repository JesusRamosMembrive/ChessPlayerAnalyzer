import torch
import torch.nn as nn

def build_model(_args=None):
    return nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
