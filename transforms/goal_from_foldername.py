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
        output_key: str = "description",
    ):
        self._specs = specs
        self.goal_mapping = dict(goal_mapping)
        self.current_task = current_task
        self.output_key = output_key

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
                    goal_desc = self.goal_mapping[key]
                    break

                path = path.parent
            else:
                # if we reach here, no matching folder name was found
                raise KeyError(
                    f"No matching goal text found for path {tensordict['path']}"
                )

        elif self.current_task is not None:
            # during rollout on an environment, we set the goal explicitly
            goal_desc = self.goal_mapping[self.current_task]

            # TODO: temporary fix
            # During training, data items are collated, which turns string goal
            # texts into a numpy array of strings. During rollout, collation
            # is disabled, so we need to wrap the goal text into a numpy array here
            goal_desc = np.array([goal_desc])

        else:
            raise RuntimeError(
                "Tensordict does not contain `path` and current_task is unset."
            )

        tensordict["goal", self.output_key] = goal_desc
        return tensordict


class GoalObjectsFromFolderName(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        goal_mapping: Mapping[str, Mapping[str, str]],
        current_task: str | None = None,
    ):
        self._specs = specs
        self.goal_mapping = dict(goal_mapping)
        self.current_task = current_task

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:

        if "path" in tensordict:
            # during training or eval on data, we extract the goal from the
            # folder name
            path = Path(tensordict["path"])
            folder_name = path.parent.name

            try:
                goal_dict = dict(self.goal_mapping[folder_name])
            except KeyError:
                raise KeyError(
                    f"No goal text found for folder name {folder_name} (full path is `{path}`)"
                )

        elif self.current_task is not None:
            # during rollout on an environment, we set the goal explicitly
            goal_dict = self.goal_mapping[self.current_task]

            # TODO: temporary fix
            # During training, data items are collated, which turns string goal
            # texts into a numpy array of strings. During rollout, collation
            # is disabled, so we need to wrap the goal text into a numpy array here
            # goal_text = np.array([goal_text])

        else:
            raise RuntimeError(
                "Tensordict does not contain `path` and current_task is unset."
            )

        if "goal" in tensordict:
            tensordict["goal"].update(goal_dict)
        else:
            tensordict["goal"] = goal_dict
        return tensordict
