import torch
from omegaconf import DictConfig


def configure_torch(torch_cfg: DictConfig) -> None:
    torch_cfg = torch_cfg or {}
    if (precision := torch_cfg.get("set_float32_matmul_precision")) is not None:
        torch.set_float32_matmul_precision(precision)
