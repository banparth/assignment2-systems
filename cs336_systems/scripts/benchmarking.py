from dataclasses import asdict, dataclass
from enum import Enum

import argparse
import statistics
import timeit
import torch.cuda.nvtx as nvtx
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.conf.conf import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_VOCAB_SIZE,
    ModelSize,
    get_arch_config,
)


@dataclass
class Config:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int


class BenchmarkMode(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"


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
        default=BenchmarkMode.FORWARD,
    )
    return parser


def resolve_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Config:
    if args.model_size is not None:
        arch = get_arch_config(args.model_size)
        return Config(
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

    return Config(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = resolve_config(args, parser)

    device = torch.device("cuda")
    model = BasicsTransformerLM(**asdict(config)).to(device=device)

    input_ids = torch.empty(DEFAULT_BATCH_SIZE, config.context_length, device=device, dtype=torch.long)
    target = torch.empty(DEFAULT_BATCH_SIZE, config.context_length, device=device, dtype=torch.long)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    def run_forward():
        torch.cuda.synchronize()
        with nvtx.range("model forward pass"):
            model.forward(input_ids)
        torch.cuda.synchronize()

    def run_backward():
        torch.cuda.synchronize()
        logits = model(input_ids)
        loss = cross_entropy(logits, target)
        loss.backward()
        torch.cuda.synchronize()

    def run_optimizer():
        torch.cuda.synchronize()
        logits = model(input_ids)
        loss = cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    run_fn = run_forward
    if args.mode == BenchmarkMode.BACKWARD:
        run_fn = run_backward
    elif args.mode == BenchmarkMode.OPTIMIZER:
        run_fn = run_optimizer

    if args.is_warmup:
        for _ in range(5):
            run_fn()

    elapsed = timeit.repeat(run_fn, number=1, repeat=1)
    print(f"config={config}")
    print(f"mode={args.mode.value}")
    print(f"timings={elapsed}")
    # print(f"mean={statistics.mean(elapsed):.6f}")
    # print(f"stdev={statistics.stdev(elapsed):.6f}")


if __name__ == "__main__":
    main()
