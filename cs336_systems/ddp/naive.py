import os
from cs336_systems.conf.common import RunResult
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit

import pandas as pd

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bulk_calls: bool = True):
        super().__init__()
        self.module = module
        self.bulk_calls = bulk_calls
        with torch.no_grad():
            for param in self.module.parameters():  
                dist.broadcast(param, src=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module.forward(x)
    
    def finish_gradient_synchronization(self):
        with torch.no_grad():
            if not self.bulk_calls:
                print(f"state 1: {torch.cuda.memory_allocated()}")
                for param in self.module.parameters():
                    if param.grad is None:
                        continue
                    dist.all_reduce(param.grad, dist.ReduceOp.AVG, async_op=False)
                print(f"state 2: {torch.cuda.memory_allocated()}")

            else:
                tensors = []
                for param in self.module.parameters():
                    if param.grad is None:
                        continue
                    tensors.append(param.grad)
                print(f"state 1: {torch.cuda.memory_allocated()}")
                flattened_tensor = torch._utils._flatten_dense_tensors(tensors)
                print(f"state 2: {torch.cuda.memory_allocated()}")
                dist.all_reduce(flattened_tensor, dist.ReduceOp.AVG, async_op=False)
                tensors = torch._utils._unflatten_dense_tensors(flattened_tensor, tensors)
                print(f"state 3: {torch.cuda.memory_allocated()}")

                i = 0
                for param in self.module.parameters():
                    if param.grad is None:
                        continue
                    param.grad = tensors[i]
                    i+=1
                print(f"state 4: {torch.cuda.memory_allocated()}")


            

