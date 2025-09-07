import logging
from typing import Sequence

from tensordict import TensorDict

from environments.specs import (
    CameraSpec,
    DataSpecs,
    DepthStream,
    ImageStream,
    PointMapStream,
    RGBStream,
)
from transforms.base_transform import Transform

STREAM_TYPES = {
    "rgb": RGBStream,
    "depth": DepthStream,
    "pointmap": PointMapStream,
    "ir": ImageStream,
}

log = logging.getLogger(__name__)


class CleanupTransform(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        keep_stream_types: str | Sequence[str] | None = None,
        drop_stream_types: str | Sequence[str] | None = None,
        keep_stream_names: str | Sequence[str] | None = None,
        drop_stream_names: str | Sequence[str] | None = None,
        drop_keys: str | Sequence[str] | None = None,
        drop_streams: str | Sequence[str] | None = None,
    ) -> None:

        all_streams = {
            (key, name): stream
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            for name, stream in spec.streams.items()
        }

        if keep_stream_types is not None:
            if drop_stream_types is not None:
                raise ValueError("Cannot specify both keep and remove streams/types")

            if isinstance(keep_stream_types, str):
                keep_stream_types = [keep_stream_types]
            else:
                keep_stream_types = list(keep_stream_types)

            keep_types = tuple(
                STREAM_TYPES[stream_type.lower()] for stream_type in keep_stream_types
            )

            self.pop_streams = {
                keys: stream
                for keys, stream in all_streams.items()
                # exact type match because ImageStream is the base class for others
                if type(stream) not in keep_types
            }

        elif drop_stream_types is not None:
            if isinstance(drop_stream_types, str):
                drop_stream_types = [drop_stream_types]
            else:
                drop_stream_types = list(drop_stream_types)

            drop_types = tuple(
                STREAM_TYPES[stream_type.lower()] for stream_type in drop_stream_types
            )

            self.pop_streams = {
                keys: stream
                for keys, stream in all_streams.items()
                # exact type match because ImageStream is the base class for others
                if type(stream) in drop_types
            }
        else:
            self.pop_streams = {}

        if keep_stream_names is not None:
            if drop_stream_names is not None:
                raise ValueError("Cannot specify both keep and remove streams/types")

            if isinstance(keep_stream_names, str):
                keep_stream_names = [keep_stream_names]
            else:
                keep_stream_names = list(keep_stream_names)

            # remove any streams that the user wants to keep by name
            self.pop_streams = {
                (key, name): stream
                for (key, name), stream in self.pop_streams.items()
                if name not in keep_stream_names
            }

        elif drop_stream_names is not None:
            if isinstance(drop_stream_names, str):
                drop_stream_names = [drop_stream_names]
            else:
                drop_stream_names = list(drop_stream_names)

            # add any streams that the user wants to drop by name
            # (duplicates don't matter, dict keys are unique)
            for (key, name), stream in all_streams.items():
                if name in drop_stream_names:
                    self.pop_streams[(key, name)] = stream

        self.pop_keys = (
            ([drop_keys] if isinstance(drop_keys, str) else list(drop_keys))
            if drop_keys is not None
            else []
        )

        drop_streams = (
            ([drop_streams] if isinstance(drop_streams, str) else list(drop_streams))
            if drop_streams is not None
            else []
        )
        for stream in drop_streams:
            key, name = stream.split("/", 1)
            if (key, name) in all_streams:
                self.pop_streams[(key, name)] = all_streams[(key, name)]
            else:
                log.warning(f"Stream {stream} not found in specs, cannot drop it.")

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key in list(obs_specs.keys()):
            # avoid modifying the dict while iterating over it
            spec = obs_specs[key]
            if key in self.pop_keys:
                obs_specs.pop(key)
                continue

            if not isinstance(spec, CameraSpec):
                continue

            streams = dict(spec.streams)
            for name in list(streams.keys()):
                if (key, name) in self.pop_streams:
                    streams.pop(name)

            if streams:
                obs_specs[key] = spec.replace(streams=streams)
            else:
                self.pop_streams = {
                    (k, n): s for (k, n), s in self.pop_streams.items() if k != key
                }
                self.pop_keys.append(key)
                obs_specs.pop(key)
        self._output_specs = specs.replace(obs=obs_specs)

        if self.pop_keys:
            log.warning(f"Dropping the following keys: {self.pop_keys}")
        if self.pop_streams:
            log.warning(
                f"Dropping the following streams: {[f'{key}/{name}' for key, name in self.pop_streams.keys()]}"
            )

    def __repr__(self) -> str:

        summary = []
        if self.pop_streams:
            summary.append(
                f"pop_streams={[f'{key}/{name}' for key, name in self.pop_streams.keys()]}"
            )

        if self.pop_keys:
            summary.append(f"pop_keys={self.pop_keys}")
        if summary:
            return f"{self.__class__.__name__}(" + ", ".join(summary) + ")"
        else:
            return f"{self.__class__.__name__}()"

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        for key in self.pop_keys:
            keys = ("obs", key)
            # use pop with default to avoid key errors
            tensordict.pop(keys, None)

        for key, name in self.pop_streams:
            keys = ("obs", key, name)
            # use pop with default to avoid key errors
            tensordict.pop(keys, None)
        return tensordict
