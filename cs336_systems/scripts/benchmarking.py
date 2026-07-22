from contextlib import nullcontext
from dataclasses import asdict, dataclass
from enum import Enum

import argparse
import statistics
import string
import timeit
from pytest import param
from sympy.polys.polyoptions import Auto
import torch.cuda.nvtx as nvtx
import torch
from torch.utils import data
import pandas as pd
from triton.testing import Benchmark

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.conf.conf import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_VOCAB_SIZE,
    ModelSize,
    get_arch_config,
    df as confs,
    ResolvedConfig,
    RunConfig,
)
from cs336_systems.conf.common import RunResult, BenchmarkMode, AutoCast
        

RAW_CONFIG_FIELDS = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=ModelSize,
        choices=list(ModelSize),
        help="Use a predefined model size from the assignment table.",
    )
    parser.add_argument("--vocab_size", type=int, help="Vocabulary size (required with raw config).")
    parser.add_argument("--context_length", type=int, help="Context length (required with raw config).")
    parser.add_argument("--d_model", type=int, help="Model width (required with raw config).")
    parser.add_argument("--num_layers", type=int, help="Number of layers (required with raw config).")
    parser.add_argument("--num_heads", type=int, help="Number of attention heads (required with raw config).")
    parser.add_argument("--d_ff", type=int, help="Feed-forward hidden size (required with raw config).")
    parser.add_argument("--is_warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mode",
        type=BenchmarkMode,
        choices=list(BenchmarkMode),
    )
    parser.add_argument(
        "--full_run",
        help="this switches the mode for a full run and treat other variable as filter",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--cast",
        type=AutoCast,
        choices=list(AutoCast)
    )
    parser.add_argument(
        "--memory_dump",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None
    )
    return parser


def resolve_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> ResolvedConfig:
    if args.model_size is not None:
        arch = get_arch_config(args.model_size)
        return ResolvedConfig(
            vocab_size=args.vocab_size if args.vocab_size is not None else DEFAULT_VOCAB_SIZE,
            context_length=args.context_length if args.context_length is not None else DEFAULT_CONTEXT_LENGTH,
            **arch,
        )

    missing = [field for field in RAW_CONFIG_FIELDS if getattr(args, field) is None]
    if missing:
        parser.error(
            "Either pass --model_size or provide all raw hyperparameters: "
            + ", ".join(f"--{field}" for field in RAW_CONFIG_FIELDS)
        )

    return ResolvedConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    )

    
def runner2(config: RunConfig) -> RunResult:
    res = config.resolved_config()
    device=torch.device("cuda")
    model = BasicsTransformerLM(**asdict(config)).to(device=device)
    input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device, dtype=torch.long)
    assert input_ids.min() >= 0 and input_ids.max() < config.vocab_size

    target = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device, dtype=torch.long)
    assert target.min() >= 0 and target.max() < config.vocab_size
    optimizer = AdamW(model.parameters(), lr=1e-3)
    return runner(model, input_ids, target, config.mode, optimizer, config.cast, device, config.name(), config.is_warmup)

def runner(model: torch.nn.Module, 
           input_ids: torch.Tensor,
           target: torch.Tensor, 
           mode: BenchmarkMode, 
           optimizer: torch.optim.Optimizer,
           cast: AutoCast,
           device: torch.device,
           memory_dump: str | None = None,
           is_warmup: bool = True,
           is_compile: bool = False) -> RunResult:
    
    if is_compile:
        model = torch.compile(model)

    def run_forward():
        torch.cuda.synchronize()
        with nvtx.range("model forward pass"):
            model.forward(input_ids)
        torch.cuda.synchronize()

    def run_backward():
        torch.cuda.synchronize()
        with nvtx.range("model forward pass"):
            logits = model(input_ids)
        with nvtx.range("model backward pass"):
            loss = cross_entropy(logits, target)
            loss.backward()
        torch.cuda.synchronize()

    def run_optimizer():
        torch.cuda.synchronize()
        with nvtx.range("model forward pass"):
            logits = model(input_ids)
        with nvtx.range("model backward pass"):
            loss = cross_entropy(logits, target)
            loss.backward()
        with nvtx.range("model optimizer"):
            optimizer.step()
        torch.cuda.synchronize()
    
    run_fn = run_forward
    if mode == BenchmarkMode.BACKWARD:
        run_fn = run_backward
    elif mode == BenchmarkMode.OPTIMIZER:
        run_fn = run_optimizer

    ctx = nullcontext()
    if cast == AutoCast.BF16:
        ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    elif cast == AutoCast.FP16:
        ctx = torch.autocast(device_type=device.type, dtype=torch.float16)
    
    original = run_fn
    def run_fn():
        with ctx:
            original()
    
    if is_warmup:
        for _ in range(5):
            run_fn()
            
    original2 = run_fn
    if memory_dump:
        def run_fn():
            torch.cuda.memory._record_memory_history(max_entries=1000000)
            original2()
            torch.cuda.memory._dump_snapshot(f"memory_dump/memory_{memory_dump}.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)

    elapsed = timeit.repeat(run_fn, number=1, repeat=10)
    
    return RunResult(elapsed)

    
def full_run(args: argparse.Namespace):
    df = pd.DataFrame({"size": list(ModelSize)})
    
    modes = pd.DataFrame({"mode": list(BenchmarkMode)})
    
    df = df.merge(modes, how="cross")
    autocasting = pd.DataFrame({"cast": list(AutoCast)})
    
    df = df.merge(autocasting, how="cross")
    compile = pd.DataFrame({"compile": list([True, False])})
    
    df = df.merge(compile, "cross")
    
    if args.model_size:
        df = df.loc[df["size"].isin([args.model_size])]

    if args.mode:
        df = df.loc[df["mode"].isin([args.mode])]
        
    if args.cast:
        df = df.loc[df["cast"].isin([args.cast])]
    
    if args.compile is not None:
        df = df.loc[df["compile"].isin([args.compile])]
    
    df["mean"] = pd.Series(dtype=object)
    df["var"] = pd.Series(dtype=object)
    for idx, row in df.iterrows():
        print(f"running : {row.to_markdown()}")

        config = ResolvedConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            context_length=args.context_length if args.context_length else DEFAULT_CONTEXT_LENGTH,
            **get_arch_config(row["size"]),
        )
        df.at[idx, "context_length"] = config.context_length
        
        try: 
            device=torch.device("cuda")
            model = BasicsTransformerLM(**asdict(config)).to(device=device)
            input_ids = torch.randint(0, config.vocab_size, (DEFAULT_BATCH_SIZE, config.context_length), device=device, dtype=torch.long)
            assert input_ids.min() >= 0 and input_ids.max() < config.vocab_size

            target = torch.randint(0, config.vocab_size, (DEFAULT_BATCH_SIZE, config.context_length), device=device, dtype=torch.long)
            assert target.min() >= 0 and target.max() < config.vocab_size
            optimizer = AdamW(model.parameters(), lr=1e-3)
            memory_file = None
            if args.memory_dump:
                memory_file = f"{row["size"]}-{row["mode"]}-{row["cast"]}-{df.at[idx, "context_length"]}"
            result = runner(model, input_ids, target, row["mode"], optimizer, row["cast"], device, memory_file, args.is_warmup, row["compile"])
            df.at[idx, "mean"] = result.mean
            df.at[idx, "var"] = result.stddev
        except torch.cuda.OutOfMemoryError:
            df.at[idx, "mean"] = "oom"
            df.at[idx, "var"] = "oom"
        
        
    print(df.to_markdown())
    
    


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if args.full_run:
        full_run(args)
    else:
        config = resolve_config(args, parser)
        device = torch.device("cuda")
        model = BasicsTransformerLM(**asdict(config)).to(device=device)

        input_ids = torch.empty(DEFAULT_BATCH_SIZE, config.context_length, device=device, dtype=torch.long)
        target = torch.empty(DEFAULT_BATCH_SIZE, config.context_length, device=device, dtype=torch.long)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        result = runner(model, input_ids, target, args.mode, optimizer, args.cast, device, args.memory_dump, args.is_warmup)
        
        print(f"config={config}")
        print(f"mode={args.mode.value}")
        print(f"timings={result.elapsed}")
        print(f"mean={result.mean:.6f}")
        print(f"stdev={result.stddev:.6f}")

    

    


if __name__ == "__main__":
    main()
