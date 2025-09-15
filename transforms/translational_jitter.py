from __future__ import annotations
from typing import Sequence, Union

from networkx import sigma
import torch
from tensordict import TensorDict

from environments.specs import CameraSpec, DataSpecs, DepthStream, RGBStream
from transforms.base_transform import Transform


class BaseJitter(Transform):
    def __init__(
        self,
        specs: DataSpecs,
    ):
        super().__init__(specs)
        
        # We only apply this jitter to 3D data streams like
        # pointmaps, depthmaps. For RGB use color jitter
        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            and any(not isinstance(stream, RGBStream) for stream in spec.streams.values())
        }
        self._input_specs = input_specs
        self._specs = self._input_specs
    
    @property
    def sigma(self) -> Union[float, int, Sequence[Union[float, int]]]:
        raise NotImplementedError
    
    @property
    def specs(self) -> DataSpecs:
        return self._specs
    
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        default_float_dtype = torch.get_default_dtype()

        for key, spec in self._input_specs.items():
            images = tensordict["obs", key]
            for name, stream in spec.streams.items():
                if not isinstance(stream, DepthStream):
                    continue

                image = images[name]

                if image.dtype != default_float_dtype:
                    image = image.to(dtype=default_float_dtype)

                image += (
                    torch.randn_like(image) * self.sigma
                )  # apply gaussian noise

                if isinstance(stream, DepthStream):
                    # clamp depth to be non-negative
                    image = torch.clamp(image, min=0.0)

                images[name] = image

        return tensordict

class TranslationalJitter(BaseJitter):
    def __init__(
        self,
        specs: DataSpecs,
        sigma: Union[float, int, Sequence[Union[float, int]]],
    ) -> None:
        super().__init__(specs)
        self._sigma = sigma

    @property
    def sigma(self) -> Union[float, int, Sequence[Union[float, int]]]:
        return self._sigma

class VariableTranslationalJitter(BaseJitter):
    def __init__(
        self,
        specs: DataSpecs,
        max_sigma: Union[float, int, Sequence[Union[float, int]]],
    ) -> None:
        super().__init__(specs)
        self._max_sigma = max_sigma

    @property
    def sigma(self) -> Union[float, int, Sequence[Union[float, int]]]:
        return torch.rand(1).item() * self._max_sigma