import functools

import numpy as np
from tensordict import TensorDict

from agents.encoders.clip_lang_encoder import LangClip
from environments.specs import DataSpecs, EmbedSpec
from transforms.base_transform import Transform


class ClipGoalEmbedding(Transform):
    def __init__(self, specs: DataSpecs):
        self.make_model()

        goal_specs = specs.goal or {}
        goal_specs = dict(goal_specs)  # copy goal specs for local modification
        goal_specs["embed"] = EmbedSpec(embed_dim=1024, n_tokens=1)
        self._output_specs = specs.replace(goal=goal_specs)

    def make_model(self):
        self.clip_model = LangClip(freeze_backbone=True, model_name="RN50")
        self.clip_model.eval()

        @functools.cache
        def cached_clip(text):
            return self.clip_model([text])

        self.cached_clip = cached_clip

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __getstate__(self):
        """Custom pickle method - exclude engine and context."""
        state = self.__dict__.copy()
        # Remove the unpicklable entries
        state.pop("cached_clip")
        state.pop("clip_model")
        return state

    def __setstate__(self, state):
        """Custom unpickle method - restore state without engine."""
        self.__dict__.update(state)
        self.make_model()

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        goal_texts = tensordict["goal", "text"]
        assert isinstance(goal_texts, np.ndarray)

        if isinstance(goal_texts[0], np.str_):
            goal_texts = [text.item() for text in goal_texts]
        elif isinstance(goal_texts[0], np.bytes_):
            goal_texts = [text.decode("utf-8") for text in goal_texts]
        else:
            raise TypeError(f"Unsupported goal text type: {type(goal_texts[0])}")

        # embedding: (B, 1, 1024)
        embedding = self.clip_model(goal_texts)
        assert embedding.shape[-2:] == (1, 1024)

        tensordict["goal", "embed"] = embedding
        return tensordict
