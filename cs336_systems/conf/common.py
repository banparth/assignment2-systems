from dataclasses import dataclass
import statistics
from enum import Enum


@dataclass
class RunResult:
    elapsed: list[float]
    
    @property
    def mean(self) -> float:
        if len(self.elapsed) == 1:
            return self.elapsed[0]
        return statistics.mean(self.elapsed)

    @property
    def stddev(self) -> float:
        if len(self.elapsed) == 1:
            return 0.0
        return statistics.stdev(self.elapsed)


class BenchmarkMode(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"
    
class AutoCast(str, Enum):
    NONE = "none"
    BF16 = "bf16"
    FP16 = "fp16"

