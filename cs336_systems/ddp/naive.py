import os
from cs336_systems.conf.common import RunResult
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit

import pandas as pd

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        with torch.no_grad():
            for param in self.module.parameters():  
                dist.broadcast(param, src=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module.forward(x)
    
    def finish_gradient_synchronization(self):
        with torch.no_grad():
            for param in self.module.parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, dist.ReduceOp.AVG, async_op=False)
            

