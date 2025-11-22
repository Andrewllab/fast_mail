from pathlib import Path
from typing import Mapping

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
            path = Path(tensordict["path"])
            folder_name = path.parent.name

            try:
                goal_text = self.goal_mapping[folder_name]
            except KeyError:
                raise KeyError(
                    f"No goal text found for folder name {folder_name} (full path is `{path}`)"
                )

        elif self.current_task is not None:
            goal_text = self.goal_mapping[self.current_task]

        else:
            raise RuntimeError(
                "Tensordict does not contain `path` and current_task is unset."
            )

        tensordict["goal", self.key] = goal_text
        return tensordict
