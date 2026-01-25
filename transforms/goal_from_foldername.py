from pathlib import Path
from typing import Mapping

import numpy as np
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform


class GoalTextFromFolderName(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        goal_mapping: Mapping[str, str],
        current_task: str | None = None,
        goal_text_key: str = "text",
    ):
        self._specs = specs
        self.goal_mapping = dict(goal_mapping)
        self.current_task = current_task
        self.key = goal_text_key

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:

        if "path" in tensordict:
            # during training or eval on data, we extract the goal from the
            # folder name
            path = Path(tensordict["path"]).with_suffix("")  # remove suffix

            while len(path.parts) > 0:
                key = "_".join(path.parts)

                if key in self.goal_mapping:
                    goal_text = self.goal_mapping[key]
                    break

                path = path.parent
            else:
                # if we reach here, no matching folder name was found
                raise KeyError(
                    f"No matching goal text found for path {tensordict['path']}"
                )

        elif self.current_task is not None:
            # during rollout on an environment, we set the goal explicitly
            goal_text = self.goal_mapping[self.current_task]

            # TODO: temporary fix
            # During training, data items are collated, which turns string goal
            # texts into a numpy array of strings. During rollout, collation
            # is disabled, so we need to wrap the goal text into a numpy array here
            goal_text = np.array([goal_text])

        else:
            raise RuntimeError(
                "Tensordict does not contain `path` and current_task is unset."
            )

        tensordict["goal", self.key] = goal_text
        return tensordict
