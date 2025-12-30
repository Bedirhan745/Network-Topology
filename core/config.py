from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    # CSV format
    sep: str = ";"
    decimal_comma: bool = True

    # Layout cache
    layout_seed: int = 42
    layout_k: float = 0.18
    layout_iterations: int = 200

    # Output paths
    outputs_dir: str = "outputs"
    layout_cache_name: str = "layout_pos_seed42.json"

