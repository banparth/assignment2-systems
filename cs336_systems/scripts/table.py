from pandas.core.frame import DataFrame


import pandas as pd


df = pd.DataFrame(
    [
        ["small", 768, 3072, 12, 12],
        ["medium", 1024, 4096, 24, 16],
        ["large", 1280, 5120, 36, 20],
        ["xl", 2560, 10240, 32, 32],
        ["10B", 4608, 12288, 50, 36]
    ],
    columns=["size", "d_model", "d_ff", "num_layers", "num_heads"]
)
