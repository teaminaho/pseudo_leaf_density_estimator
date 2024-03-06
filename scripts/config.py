import toml
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class LDIConfig:
    lch_lower: List[float]
    lch_upper: List[float]
    hsv_lower: List[int]
    hsv_upper: List[int]
    grid_size: Optional[int] = None
    kernel_size: Optional[int] = None
    density_min: Optional[int] = None
    num_density_bins: Optional[int] = None
    divided_area: Optional[List[int]] = None
    k_h_size: Optional[int] = None
    serch_area: Optional[List[int]] = None
    horizontal_crop_ratio_list: Optional[List[float]] = None
    border_color: Optional[List[List[int]]] = None

    @staticmethod
    def from_toml(config_path: str) -> "LDIConfig":
        with open(config_path, "r") as f:
            data = toml.load(f)
        return LDIConfig(**data)
