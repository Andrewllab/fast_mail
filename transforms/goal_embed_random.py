import logging

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from environments.specs import DataSpecs, EmbedSpec
from transforms.base_transform import NormalizingTransform

log = logging.getLogger(__name__)


class RandomGoalEmbedding(NormalizingTransform, nn.Module):
    def __init__(self, specs: DataSpecs, embed_dim: int = 1024, text_key: str = "text"):
        super().__init__()

        goal_specs = specs.goal or {}
        goal_specs = dict(goal_specs)  # copy goal specs for local modification
        goal_specs["embed"] = EmbedSpec(embed_dim=embed_dim, n_tokens=1)
        self._output_specs = specs.replace(goal=goal_specs)

        self.text_key = text_key
        self.goal_texts = {}
        self.model = None

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def make_model(self):
        log.info(
            f"Creating random goal embedding model for {len(self.goal_texts)} unique goals."
        )
        self.goal_texts = {text: idx for idx, text in enumerate(self.goal_texts.keys())}
        self.model = nn.Embedding(
            len(self.goal_texts), self._output_specs.goal["embed"].embed_dim
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None  # do not pickle the model
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self.model is None:
            # Create the embedding model once we have seen all goals.
            # Here we rely on the fact that the transform will always be
            # pickled after preprocessing (in prepare_data) and unpickled
            # before training begins (in setup).
            # TODO: add a hook to inform transforms when preprocessing is done
            self.make_model()

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        # a trajectory should only have a single goal
        goal_text = tensordict["goal", self.text_key]

        # we use the dictionary like a set just to store unique texts
        self.goal_texts[goal_text] = 0

        return tensordict

    def forward(self, tensordict: TensorDict) -> TensorDict:
        goal_texts = tensordict["goal", "text"]
        assert isinstance(goal_texts, np.ndarray)

        if isinstance(goal_texts[0], np.str_):
            goal_texts = [text.item() for text in goal_texts]
        elif isinstance(goal_texts[0], np.bytes_):
            goal_texts = [text.decode("utf-8") for text in goal_texts]
        else:
            raise TypeError(f"Unsupported goal text type: {type(goal_texts[0])}")

        idxs = [self.goal_texts[text] for text in goal_texts]
        idxs_tensor = torch.tensor(idxs, dtype=torch.long, device=tensordict.device)
        # embedding: (B, 1, 1024)
        assert self.model is not None
        embedding = self.model(idxs_tensor).unsqueeze(dim=1)
        assert embedding.shape[-2:] == (1, self._output_specs.goal["embed"].embed_dim)

        tensordict["goal", "embed"] = embedding
        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        # we do not need to reverse the embedding
        return tensordict
