"""Batched-camera RGB-D 3D replay entry point.

This delegates to the shared CLI but selects ``alex_batch_3d``'s single
batched-CoTracker implementation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alex.simulate_static_3d import run


if __name__ == "__main__":
    run(batched=True)
