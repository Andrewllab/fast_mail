import functools

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
        goal_text = tensordict["goal", "text"]

        goal_text = str(goal_text)  # make sure it's a string
        assert isinstance(goal_text, str)

        embedding = self.cached_clip(goal_text)
        assert embedding.shape == (1, 1, 1024)

        tensordict["goal", "embed"] = embedding.squeeze(0)  # shape (1, 1024)

        return tensordict
