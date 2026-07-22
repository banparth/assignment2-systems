import pandas as pd
from enum import Enum
from dataclasses import dataclass
from .common import BenchmarkMode, AutoCast

DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_BATCH_SIZE = 4


class ModelSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XL = "xl"
    B10 = "10b"


df = pd.DataFrame(
    [
        [ModelSize.SMALL, 768, 3072, 12, 12],
        [ModelSize.MEDIUM, 1024, 4096, 24, 16],
        [ModelSize.LARGE, 1280, 5120, 36, 20],
        [ModelSize.XL, 2560, 10240, 32, 32],
        [ModelSize.B10, 4608, 12288, 50, 36],
    ],
    columns=["size", "d_model", "d_ff", "num_layers", "num_heads"],
).set_index("size")


def get_arch_config(model_size: ModelSize) -> dict[str, int]:
    row = df.loc[model_size]
    return {
        "d_model": int(row["d_model"]),
        "d_ff": int(row["d_ff"]),
        "num_layers": int(row["num_layers"]),
        "num_heads": int(row["num_heads"]),
    }

@dataclass
class ResolvedConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int

@dataclass
class RunConfig:
    model: ModelSize
    batch_size: int
    vocab_size: int
    context_length: int
    mode: BenchmarkMode
    cast: AutoCast
    is_memory_dump: bool
    is_warmup: bool
    
    def name(self) -> str:
        return f"{self.model}-{self.batch_size}-{self.vocab_size}-{self.context_length}-{self.mode}-{self.cast}"
    
    def resolved_config(self) -> ResolvedConfig:
        arch = get_arch_config(self.model)
        return ResolvedConfig(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            **arch,
        )
