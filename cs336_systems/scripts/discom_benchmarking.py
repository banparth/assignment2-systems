import os
from cs336_systems.conf.common import RunResult
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit

import pandas as pd

def club_run_result(results: list[RunResult]) -> RunResult:
    final: RunResult = RunResult([])
    
    for result in results:
        final.elapsed.extend(result.elapsed)
    
    return final
    
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if torch.cuda.device_count() == 1: # this heuristics is specific to my setup
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
def exit(rank: int):
    dist.destroy_process_group()
    print(f"exiting {rank=}")

def distributed_reduction(rank: int,
                          world_size: int,
                          size: int,
                          queue: mp.Queue):
    if torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
    else:
        torch.cuda.set_device(rank)
    setup(rank, world_size)
    data = torch.rand(size//4).to("cuda")
    print(f"{data.get_device()=}")
    print(f"rank {rank} data (before all-reduce): {data}")
    
    def run():
        torch.cuda.synchronize()
        dist.all_reduce(data, op=dist.ReduceOp.AVG, async_op=False)
        torch.cuda.synchronize()
        
    for _ in range(5):
        run()
        
    print(f"rank {rank} data (after warming all-reduce): {data}")
    elapsed = timeit.repeat(run, number=1, repeat=10)
    r = RunResult(elapsed)

    results = [RunResult([]) for _ in range(world_size)]
    dist.all_gather_object(results, r)
    
    
    if rank == 0:
        queue.put(club_run_result(results))
    
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
    
    # device = torch.device("cuda")
    _1mb = 1024*1024
    df = pd.DataFrame({"size": list([_1mb, 10*_1mb, 100*_1mb, 1024*_1mb])})
    
    num_gpu = pd.DataFrame({f"num_gpu": list([2, 4, 6])})
    
    df = df.merge(num_gpu, "cross")
    
    df["mean"] = pd.Series(dtype=object)
    df["var"] = pd.Series(dtype=object)
    
    for idx, row in df.iterrows():
        size = int(row["size"])
        num_gpu = int(row["num_gpu"])

        queue = mp.get_context("spawn").Queue()
        
        ctx = mp.spawn(fn=distributed_reduction, args=(num_gpu,size,queue), nprocs=num_gpu, join=False)
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
    
    
# Results from 2*GPU
# |    |       size |   num_gpu |        mean |         var |
# |---:|-----------:|----------:|------------:|------------:|
# |  0 |    1048576 |         2 | 7.14495e-05 | 4.53353e-05 |
# |  1 |   10485760 |         2 | 9.59937e-05 | 3.16923e-05 |
# |  2 |  104857600 |         2 | 0.00042573  | 2.75406e-05 |
# |  3 | 1073741824 |         2 | 0.00324733  | 2.7898e-05  |