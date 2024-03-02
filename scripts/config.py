import toml
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LDIConfig:
    lsh_lower: List[float]
    lsh_upper: List[float]
    hsv_lower: List[int]
    hsv_upper: List[int]
    grid_size: int
    kernel_size: int
    density_min: int
    num_density_bins: int
    divided_area: List[int]
    k_h_size: int
    serch_area: List[int]
    crop_rate: List[float] = None
    border_color: List[List[int]] = None

    @staticmethod
    def from_toml(config_path: str):
        with open(config_path, "r") as f:
            data = toml.load(f)
        return LDIConfig(**data)
