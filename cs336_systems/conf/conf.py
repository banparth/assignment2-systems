import pandas as pd
from enum import Enum

DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_BATCH_SIZE = 4


class ModelSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XL = "xl"
    B10 = "10b"


_df = pd.DataFrame(
    [
        [ModelSize.SMALL, 768, 3072, 12, 12],
        [ModelSize.MEDIUM, 1024, 4096, 24, 16],
        [ModelSize.LARGE, 1280, 5120, 36, 20],
        [ModelSize.XL, 2560, 10240, 32, 32],
        [ModelSize.B10, 4608, 12288, 50, 36],
    ],
    columns=["size", "d_model", "d_ff", "num_layers", "num_heads"],
)

df = _df.set_index("size")


def get_arch_config(model_size: ModelSize) -> dict[str, int]:
    row = df.loc[model_size]
    return {
        "d_model": int(row["d_model"]),
        "d_ff": int(row["d_ff"]),
        "num_layers": int(row["num_layers"]),
        "num_heads": int(row["num_heads"]),
    }
