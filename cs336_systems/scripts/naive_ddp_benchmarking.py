import os
from cs336_systems.conf.common import RunResult
from cs336_systems.conf.conf import (ModelSize, 
                                get_arch_config, 
                                DEFAULT_BATCH_SIZE, 
                                DEFAULT_CONTEXT_LENGTH,
                                DEFAULT_VOCAB_SIZE,
                                ResolvedConfig)
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp.naive import DDP
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.optimizer import AdamW
import argparse
import timeit
import pandas as pd
from dataclasses import asdict
from typing import Literal

def club_run_result(results: list[RunResult]) -> RunResult:
    final: RunResult = RunResult([])
    
    for result in results:
        final.elapsed.extend(result.elapsed)
    
    return final
    
def setup(rank, world_size):
    print(f"started to setting up: {rank=}")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if torch.cuda.device_count() == 1: # this heuristics is specific to my setup
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        print(f"started to setting up2: {rank=}, {world_size=}")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
def exit(rank: int):
    dist.destroy_process_group()
    print(f"exiting {rank=}")

def distributed_reduction(rank: int,
                          world_size: int,
                          config: ResolvedConfig,
                          mode: Literal["full", "comms"],
                          queue: mp.Queue):
    if torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
    else:
        torch.cuda.set_device(rank)
    setup(rank, world_size)
    print(f"setting up {rank=}")
    try:
        device = torch.device("cuda")
        model = BasicsTransformerLM(**asdict(config))
        model = model.to(device)

        
        input_ids = torch.randint(0, config.vocab_size, (DEFAULT_BATCH_SIZE//world_size, config.context_length), device=device, dtype=torch.long)
        assert input_ids.min() >= 0 and input_ids.max() < config.vocab_size
        model = DDP(model)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        
        def run():
            torch.cuda.synchronize()
            y = model.forward(input_ids)
            y.sum().backward()
            model.finish_gradient_synchronization()
            optimizer.step()
            torch.cuda.synchronize()
            
        def run_comms():
            torch.cuda.synchronize()
            model.finish_gradient_synchronization()
            torch.cuda.synchronize()
            
        for _ in range(5):
            run()
            
        if mode == "comms":
            run = run_comms
            
            
        elapsed = timeit.repeat(run, number=1, repeat=10)
        r = RunResult(elapsed)

        results = [RunResult([]) for _ in range(world_size)]
        dist.all_gather_object(results, r)
        
        
        if rank == 0:
            queue.put(club_run_result(results))
    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            queue.put(RunResult([]))
    
    exit(rank)
    
    
    
# if __name__ == "__main__":
#     world_size = 4
#     mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()
    world_size = 2
    
    
    df = pd.DataFrame({"size": list([ModelSize.XL])})
    mode = pd.DataFrame({"mode": list(["full", "comms"])})
    
    df = df.merge(mode, "cross")
    
    
    
    df["mean"] = pd.Series(dtype=object)
    df["var"] = pd.Series(dtype=object)
    
    for idx, row in df.iterrows():
        config = ResolvedConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            context_length= DEFAULT_CONTEXT_LENGTH,
            **get_arch_config(row["size"]),
        )

        mode = row["mode"]
        print(df.loc[idx].to_markdown())

        queue = mp.get_context("spawn").Queue()
        num_gpu = 2
        ctx = mp.spawn(fn=distributed_reduction, args=(num_gpu,config,mode, queue,), nprocs=num_gpu, join=False)
        assert ctx is not None
        result: RunResult = queue.get()
        
        while not ctx.join():
            pass
                
        df.at[idx, "mean"] = result.mean
        df.at[idx, "var"] = result.stddev
        print(df.loc[idx].to_markdown())
    
    print(df.to_markdown())


if __name__ == "__main__":
    main()
    