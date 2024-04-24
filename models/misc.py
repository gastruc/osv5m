import pandas as pd
import torch
from torch import nn


class DoNothingOptimizer(nn.Module):
    def __init__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def zero_grad(self, *args, **kwargs):
        pass
