from torch._C import dtype


from torch._tensor import Tensor


from typing import Any, Callable


import argparse
import pandas as pd
import torch
from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.attention.fa2_triton import FA2TritonFunc
from cs336_systems.conf.common import RunResult, BenchmarkMode
import timeit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int, help="Context length (required with raw config).")
    parser.add_argument("--is_warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--full_run",
        help="this switches the mode for a full run and treat other variable as filter",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--memory_dump",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()
    
    device = torch.device("cuda")
    df = pd.DataFrame({"d_model": list([16, 32, 64, 128])})
    seq = pd.DataFrame({"seq_len": list([128, 256, 1024, 4096, 8192, 16384, 32768, 65536])})
    precision = pd.DataFrame({"precision": list([torch.float32])})
    kernel = pd.DataFrame({"kernel": list([scaled_dot_product_attention, 
                                           FA2TritonFunc().apply, 
                                           torch.compile(scaled_dot_product_attention, fullgraph=True)])})
    mode = pd.DataFrame({"mode": list([BenchmarkMode.FORWARD, BenchmarkMode.BACKWARD])})
    
    df = df.merge(seq, "cross")
    df = df.merge(precision, "cross")
    df = df.merge(kernel, "cross")
    df = df.merge(mode, "cross")
    
    
    df["mean"] = pd.Series(dtype=object)
    df["var"] = pd.Series(dtype=object)
    # df["mem"] = pd.Series(dtype=object)
    # df["memb"] = pd.Series(dtype=object)
    
    for idx, row in df.iterrows():
        seq_len = int(row["seq_len"])
        d_model = int(row["d_model"])
        precision: dtype = row["precision"]
        kernel = row["kernel"]
        mode = BenchmarkMode(row["mode"])

        q = torch.randn((1, seq_len, d_model), device=device, requires_grad=True, dtype=precision)
        k = torch.randn((1, seq_len, d_model), device=device, requires_grad=True, dtype=precision)
        v = torch.randn((1, seq_len, d_model), device=device, requires_grad=True, dtype=precision)
        a: int = 0
        b: int = 0
        func: Callable[..., Tensor] = kernel
        
        def run_forward():
            torch.cuda.synchronize()
            func(q, k, v)
            torch.cuda.synchronize()

        def run_backward():
            torch.cuda.synchronize()
            y = func(q, k, v)
            y.sum().backward()
            torch.cuda.synchronize()
            
        run = run_forward
        if mode == BenchmarkMode.BACKWARD:
            run = run_backward


        try:
            for _ in range(5):
                run()

            elapsed = timeit.repeat(run, number=1, repeat=10)
            r = RunResult(elapsed)
            df.at[idx, "mean"] = r.mean
            df.at[idx, "var"] = r.stddev
            # df.at[idx, "mem"] = a
            # df.at[idx, "memb"] = b
        except torch.cuda.OutOfMemoryError:
            df.at[idx, "mean"] = "oom"
            df.at[idx, "var"] = "oom"
            # df.at[idx, "mem"] = "oom"
            # df.at[idx, "memb"] = "oom"

        
        print(df.loc[idx].to_markdown())
            
        
        
    
    print(df.to_markdown())


if __name__ == "__main__":
    main()