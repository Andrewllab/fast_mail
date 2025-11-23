from __future__ import annotations

import logging

import torch
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform

log = logging.getLogger(__name__)


class BinarizeGripperActions(Transform):
    """Binarizes gripper actions that represent a target width for the gripper.
    Gripper actions are mapped to 0 or 2*threshold if they are below or above
    the threshold, respectively.

    The ref_action is modified too, since this transform destroys information
    and there is no way for the model to know what the original action was.
    There is no need to reverse this transform since the binarized target width
    can be sent to the environment as-is.

    Args:
        specs (DataSpecs): data specs
        threshold (float | None): threshold for binarization. If None, half the
            maximum gripper width in the trajectory is used.
        open_width (float): width to set when gripper is open.
        hysteris_steps (int): number of steps to use for hysteresis check. If
            there are state changes that happen more frequently than this number
            of steps, a warning is logged.
    """

    def __init__(
        self,
        specs: DataSpecs,
        threshold: float | None = None,
        open_width: float = 0.08,
        hysteris_steps: int = 30,
    ):
        self.threshold = threshold
        self.open_width = open_width
        self.hysteris_steps = hysteris_steps
        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # when called during eval, there is no action to convert
        if "action" in tensordict:
            gripper_action = tensordict["action"][..., -1]

            if self.threshold is None:
                threshold = (gripper_action.max() / 2).item()
            else:
                threshold = self.threshold

            closed = gripper_action < threshold
            state_changes = closed[1:] ^ closed[:-1]
            log.debug(
                "Trajectory %sfrom %s has %s changes of state of the gripper.",
                (f"named {tensordict['name']} " if "name" in tensordict else ""),
                tensordict["path"],
                state_changes.sum().item(),
            )

            state_change_idxs = state_changes.nonzero(as_tuple=True)[0]
            times_between_state_changes = state_change_idxs[1:] - state_change_idxs[:-1]
            if (times_between_state_changes < self.hysteris_steps).any():
                idx = (
                    (times_between_state_changes < self.hysteris_steps)
                    .to(torch.int32)
                    .argmax()
                )
                start = state_change_idxs[idx]
                end = state_change_idxs[idx + 1]
                log.warning(
                    "Trajectory %sfrom %s has a change of gripper state that lasts <%s time steps (between time %s and %s)",
                    (f"named {tensordict['name']} " if "name" in tensordict else ""),
                    tensordict["path"],
                    self.hysteris_steps,
                    start,
                    end,
                )

            gripper_action[closed] = 0.0
            gripper_action[~closed] = self.open_width

            if "ref_action" in tensordict:
                tensordict["ref_action"][..., -1] = gripper_action

        return tensordict
