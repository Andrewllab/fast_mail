import logging
import random

import numpy as np
import torch

log = logging.getLogger(__name__)


def get_rng(cfg):
    seed = cfg.get("seed")
    rng = np.random.default_rng(seed=seed)
    if hasattr(rng.bit_generator, "seed_seq"):
        entropy = rng.bit_generator.seed_seq.entropy
    else:
        # older versions of numpy
        entropy = rng.bit_generator._seed_seq.entropy
    if seed is None:
        log.info(f"Using random seed {entropy} for this run")

    return rng


def manual_seed(rng):
    torch.manual_seed(rng.integers(np.iinfo(np.int64).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng.integers(np.iinfo(np.int64).max))
    np.random.seed(rng.integers(np.iinfo(np.uint32).max))
    random.seed(rng.integers(np.iinfo(np.int64).max))
