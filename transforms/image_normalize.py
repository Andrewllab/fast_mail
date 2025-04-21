import torch
import torchvision.transforms.functional as F
from tensordict import TensorDict

from environments.specs import CameraSpec, DataSpecs, RGBStream
from transforms.base_transform import Transform


class NormalizeImage(Transform):
    def __init__(self, specs: DataSpecs, mean, std) -> None:

        self.mean = mean
        self.std = std

        # find the specs that this transform acts on
        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            and any(isinstance(stream, RGBStream) for stream in spec.streams.values())
        }
        self._input_specs = input_specs

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key, spec in input_specs.items():
            streams = dict(spec.streams)  # copy streams for local modification
            for name, stream in streams.items():
                stream = stream.reorder_channels("CHW")
                streams[name] = stream
            obs_specs[key] = spec.replace(streams=streams)
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        default_float_dtype = torch.get_default_dtype()

        for key, spec in self._input_specs.items():
            images = tensordict["obs", key]
            for name, stream in spec.streams.items():
                if not isinstance(stream, RGBStream):
                    continue

                image = images[name]

                if stream.channel_order == "HWC":
                    image = torch.movedim(image, -1, -3)

                if image.dtype != default_float_dtype:
                    image = image.to(dtype=default_float_dtype).div(255)

                image = F.normalize(image, self.mean, self.std, inplace=True)

                images[name] = image

        return tensordict
