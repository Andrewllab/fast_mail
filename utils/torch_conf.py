import logging

import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def configure_torch(torch_cfg: DictConfig | None) -> None:
    torch_cfg = torch_cfg or {}
    if (precision := torch_cfg.get("set_float32_matmul_precision")) is not None:
        log.debug(f"Setting precision of float32 matmul to {precision}")
        torch.set_float32_matmul_precision(precision)

    if print_opts := (torch_cfg.get("print_options", {}) or {}):
        log.debug(f"Setting print options: {print_opts}")
        torch.set_printoptions(**print_opts)
