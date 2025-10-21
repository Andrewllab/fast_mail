import dataclasses
import os.path as osp
from typing import Sequence

import torch
from tensordict import TensorDict

from environments.specs import CameraSpec, DataSpecs, DepthStream
from third_party.FoundationStereo.core.utils.utils import InputPadder
from transforms.base_transform import Transform, TransformConstraint
from utils.paths import resolve_path
from utils.tensor_rt import get_metadata, load_engine, run_inference


class FoundationStereo(Transform):

    constraints = [TransformConstraint.GPU_ONLY]

    def __init__(
        self,
        specs: DataSpecs,
        engine_path: str,
        cam_keys: str | Sequence[str] | None = None,
        remove_invisible: bool = True,
    ) -> None:

        self.remove_invisible = remove_invisible

        # Load tensorRT engine
        self.engine_path = engine_path
        self._engine, self._context = None, None

        if isinstance(cam_keys, str):
            cam_keys = [cam_keys]

        self._input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            and "left" in spec.streams
            and "right" in spec.streams
            and spec.baseline is not None
            and (cam_keys is None or key in cam_keys)
        }

        fs_metadata = get_metadata(self.engine)
        shapes = [tensor["shape"] for tensor in fs_metadata["in"].values()]
        assert all(size == shapes[0] for size in shapes)
        fs_shape = shapes[0]

        self.batch_size = fs_shape[0]
        if self.batch_size not in (len(self._input_specs), 1):
            raise ValueError(
                f"Batch size of TensorRT model {self.batch_size} must either match the number of cameras ({len(self._input_specs)}) or be 1"
            )

        # instantiate functions to pad the inputs
        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        shapes = []
        for key, spec in self._input_specs.items():
            left = spec.streams["left"]
            right = spec.streams["right"]

            if left.shape != right.shape:
                raise ValueError(
                    f"Left and right images must have the same shape, but got {left.shape} and {right.shape}"
                )
            shapes.append(left.height_width)

            depth_names = [
                name
                for name, stream in spec.streams.items()
                if isinstance(stream, DepthStream)
            ]
            # we assume one camera can have at most one depth stream
            assert len(depth_names) == 1
            depth_name = depth_names[0]

            streams = dict(spec.streams)  # copy streams for local modification
            depth_stream = DepthStream(
                height=left.height,
                width=left.width,
                time=spec.time,
                intrinsics=left.intrinsics,  # intrinsics of the virtual "depth camera" are the same as the left camera
            )
            # update stream with resized shape and modified camera intrinsics
            streams[depth_name] = depth_stream

            obs_specs[key] = dataclasses.replace(spec, streams=streams)
        self._output_specs = specs.replace(obs=obs_specs)

        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(
                f"All input images must have the same shape, but got {shapes}"
            )

        # instantiate input padder using any stream's height_width, since they're all the same
        self.padder = InputPadder(left.height_width, divis_by=32, force_square=False)

        # check if the image shape is correct after padding
        # padder only accepts inputs with 4 dimensions
        padded_left = self.padder.pad(torch.zeros((1, 1) + left.height_width))[0]
        if padded_left.shape[-2:] != fs_shape[-2:]:
            raise ValueError(
                f"Left and right images must have the same shape as the foundation stereo model, but got {left.height_width} and {fs_shape[-2:]}"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(engine_name={osp.basename(self.engine_path)})"
        )

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    @property
    def engine(self):
        # load engine lazily because not every platform has it but may need
        # tp unpickle this transform to access the specs
        if self._engine is None or self._context is None:
            self._engine, self._context = load_engine(resolve_path(self.engine_path))
        return self._engine

    @property
    def context(self):
        if self._engine is None or self._context is None:
            self._engine, self._context = load_engine(resolve_path(self.engine_path))
        return self._context

    def __getstate__(self):
        """Custom pickle method - exclude engine and context."""
        state = self.__dict__.copy()
        # Remove the unpicklable entries
        state.pop("_engine")
        state.pop("_context")
        return state

    def __setstate__(self, state):
        """Custom unpickle method - restore state without engine."""
        self.__dict__.update(state)
        self._engine, self._context = None, None

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        default_dtype = torch.get_default_dtype()

        lefts = []
        rights = []
        for key, spec in self._input_specs.items():
            left_stream = spec.streams["left"]

            left = tensordict["obs", key, "left"]
            right = tensordict["obs", key, "right"]

            assert (
                left.shape == right.shape
            ), "Left and right images must have the same shape"

            n_image_dims = left_stream.n_image_dims
            # leading dims are either (B, T) or just (T) in preprocessing
            leading_dims = left.shape[:-n_image_dims]

            # flatten batch dimensions
            left = left.flatten(end_dim=-n_image_dims - 1)  # inclusive
            right = right.flatten(end_dim=-n_image_dims - 1)

            if left_stream.channels is None:
                # expand intensity images to 3 channels (in torch order)
                left = left.unsqueeze(-3).expand(-1, 3, -1, -1)
                right = right.unsqueeze(-3).expand(-1, 3, -1, -1)

            if left_stream.channel_order == "HWC":
                # move channels to the front (in torch order)
                left = torch.movedim(left, -1, -3)
                right = torch.movedim(right, -1, -3)

            # as of this point, all images are in the shape (B, 3, H, W)
            assert len(left.shape) == len(right.shape) == 4
            assert left.shape[-3] == right.shape[-3] == 3

            # unlike everything else in fast_mail, foundation stereo expects
            # float inputs between 0 and 255
            if left.dtype == torch.uint8:
                left = left.to(default_dtype)
            else:
                assert left.max() <= 1.0
                left = left.mul(255)
            if right.dtype == torch.uint8:
                right = right.to(default_dtype)
            else:
                assert right.max() <= 1.0
                right = right.mul(255)

            lefts.append(left)
            rights.append(right)

        lefts = torch.cat(lefts, dim=0)  # (N*B, 3, H, W)
        rights = torch.cat(rights, dim=0)  # (N*B, 3, H, W)

        # pad images dimensions to be divisible by 32
        lefts, rights = self.padder.pad(lefts, rights)

        # (N*B, 3, H, W) -> (n_batches, batch_size, 3, H, W)
        # since self.batch_size is either 1 or N, we can always group into batches like this
        lefts = lefts.unflatten(dim=0, sizes=(-1, self.batch_size))
        rights = rights.unflatten(dim=0, sizes=(-1, self.batch_size))

        disps = []
        for left_batch, right_batch in zip(lefts, rights):
            disp = run_inference(
                self.engine, self.context, left=left_batch, right=right_batch
            )
            disps.append(disp)

        disps = torch.cat(disps, dim=0)  # (N*B, 1, H, W)

        disps = self.padder.unpad(disps)

        if self.remove_invisible:
            disps = remove_invisible(disps)

        # (N*B, 1, H, W) -> (N, B, H, W)
        disps = disps.squeeze(dim=1).unflatten(
            dim=0, sizes=(len(self._input_specs), -1)
        )

        for (key, spec), disp in zip(self._input_specs.items(), disps):
            depth = spec.intrinsics.fx * spec.baseline / disp

            # depth: (B, H, W) -> (B, T, H, W)
            depth = depth.unflatten(dim=0, sizes=leading_dims)

            tensordict["obs", key, "depth"] = depth

        return tensordict


def remove_invisible(disparity: torch.Tensor) -> torch.Tensor:
    """Remove invisible pixels from the disparity map.

    Modified from third_party/FoundationStereo/scripts/run_demo.py
    """

    yy, xx = torch.meshgrid(
        torch.arange(
            disparity.shape[-2], dtype=disparity.dtype, device=disparity.device
        ),
        torch.arange(
            disparity.shape[-1], dtype=disparity.dtype, device=disparity.device
        ),
        indexing="ij",
    )
    us_right = xx - disparity
    invalid = us_right < 0
    disparity[invalid] = torch.inf

    return disparity
