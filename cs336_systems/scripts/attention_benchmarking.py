import argparse
import pandas as pd
import torch
from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.conf.common import RunResult
import timeit

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()
    
    device = torch.device("cuda")
    df = pd.DataFrame({"d_model": list([16, 32, 64, 128])})
    seq = pd.DataFrame({"seq_len": list([256, 1024, 4096, 8192, 16384])})
    
    df = df.merge(seq, "cross")
    
    df["mean"] = pd.Series(dtype=object)
    df["var"] = pd.Series(dtype=object)
    df["mem"] = pd.Series(dtype=object)
    df["memb"] = pd.Series(dtype=object)
    
    for idx, row in df.iterrows():
        seq_len = int(row["seq_len"])
        d_model = int(row["d_model"])

        q = torch.randn((8, seq_len, d_model), device=device, requires_grad=True)
        k = torch.randn((8, seq_len, d_model), device=device, requires_grad=True)
        v = torch.randn((8, seq_len, d_model), device=device, requires_grad=True)
        print(q.requires_grad)
        print(k.requires_grad)
        a: int = 0
        b: int = 0
        func = torch.compile(scaled_dot_product_attention, fullgraph=True)
        def run():
            nonlocal a
            nonlocal b
            torch.cuda.synchronize()
            y = func(q, k, v)
            a = torch.cuda.memory_allocated(device)
            y.sum().backward()
            b = torch.cuda.memory_allocated(device)
            torch.cuda.synchronize()
        try:
            elapsed = timeit.repeat(run, number=1, repeat=100)
            r = RunResult(elapsed)
            df.at[idx, "mean"] = r.mean
            df.at[idx, "var"] = r.stddev
            df.at[idx, "mem"] = a
            df.at[idx, "memb"] = b
        except torch.cuda.OutOfMemoryError:
            df.at[idx, "mean"] = "oom"
            df.at[idx, "var"] = "oom"
            df.at[idx, "mem"] = "oom"
            df.at[idx, "memb"] = "oom"

        
        print(df.loc[idx].to_markdown())
            
        
        
    
    print(df.to_markdown())


if __name__ == "__main__":
    main()