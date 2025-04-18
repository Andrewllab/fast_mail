import dataclasses
import os
from collections import namedtuple

import numpy as np
import tensorrt as trt
import torch
from tensordict import TensorDict

from environments.specs import CameraSpec, DataSpecs, DepthStream
from transforms.base_transform import Transform
from utils.FoundationStereo.utils import InputPadder

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

BASELINES = {
    "gripper_cam": 0.0180272,
    "left_cam": 0.0500103,
    "right_cam": 0.0501555,
}


class ModelData:
    INPUT_NAME_0 = "left"
    INPUT_NAME_1 = "right"
    DTYPE = trt.float32


class FoundationStereo(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        engine_path: str = "/tensorRT.engine",
        remove_invisible: bool = True,
    ) -> None:

        self.remove_invisible = remove_invisible

        self._input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            and "left" in spec.streams
            and "right" in spec.streams
        }

        # instantiate functions to pad the inputs
        # create a modified specs object for the output
        self._padders = {}
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key, spec in self._input_specs.items():
            left = spec.streams["left"]
            right = spec.streams["right"]
            assert (
                left.shape == right.shape
            ), "Left and right images must have the same shape"
            self._padders[key] = InputPadder(
                left.height_width, divis_by=32, force_square=False
            )

            streams = dict(spec.streams)  # copy streams for local modification
            depth_stream = DepthStream(
                height=left.height, width=left.width, intrinsics=left.intrinsics
            )
            # update stream with resized shape and modified camera intrinsics
            streams["depth"] = depth_stream

            obs_specs[key] = dataclasses.replace(spec, streams=streams)
        self._output_specs = specs.replace(obs=obs_specs)

        # Load tensorRT engine
        self.device = torch.device("cuda")  # TODO: avoid hardcoding this
        self.engine_path = engine_path
        self.engine = load_engine(self.engine_path)
        self.bindings = allocate_bindings(self.engine, self.device)

        self.context = self.engine.create_execution_context()

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        default_dtype = torch.get_default_dtype()

        for key, spec in self._input_specs.items():
            left_stream = spec.streams["left"]

            left = tensordict["obs", key, "left"]
            right = tensordict["obs", key, "right"]

            assert (
                left.shape == right.shape
            ), "Left and right images must have the same shape"

            n_image_dims = left_stream.n_image_dims
            leading_dims = left.shape[:-n_image_dims]

            # flatten batch dimensions
            left = left.flatten(end_dim=-n_image_dims - 1)  # inclusive
            right = right.flatten(end_dim=-n_image_dims - 1)

            if left_stream.channels is None:
                # expand intensity images to 3 channels (in torch order)
                left = left.unsqueeze(-3).expand(-1, 3, -1, -1)
                right = right.unsqueeze(-3).expand(-1, 3, -1, -1)

            if left_stream.channel_order == "HWC":
                # move channels to the end (in torch order)
                left = torch.movedim(left, -1, -3)
                right = torch.movedim(right, -1, -3)

            # as of this point, all images are in the shape (B, 3, H, W)
            assert (
                left.shape[-3] == right.shape[-3] == 3
            ), "RGB images must have 3 channels"
            _, C, H, W = left.shape

            if left.dtype == torch.uint8:
                left = left.to(default_dtype)
                right = right.to(default_dtype)
            else:
                assert left.max() <= 1.0 and right.max() <= 1.0
                left = left.mul(255)
                right = right.mul(255)

            # pad images dimensions to be divisible by 32
            padder = self._padders[key]
            left, right = padder.pad(left, right)

            self.bindings[ModelData.INPUT_NAME_0] = self.bindings[
                ModelData.INPUT_NAME_0
            ]._replace(data=left, ptr=int(left.data_ptr()))
            self.bindings[ModelData.INPUT_NAME_1] = self.bindings[
                ModelData.INPUT_NAME_1
            ]._replace(data=right, ptr=int(right.data_ptr()))

            # foundation stereo
            run_inference(self.engine, self.context, self.bindings)

            output_binding = [
                name
                for name in self.bindings
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
            ][0]
            output = self.bindings[output_binding].data
            disp = output.reshape(self.bindings[output_binding].shape)

            disp = padder.unpad(disp)
            disp = disp.data.reshape(H, W)

            if self.remove_invisible:
                yy, xx = torch.meshgrid(
                    torch.arange(disp.shape[0]),
                    torch.arange(disp.shape[1]),
                    indexing="ij",
                )
                xx = xx.to(disp.device)
                us_right = xx - disp
                invalid = us_right < 0
                disp[invalid] = np.inf

            baseline = BASELINES[key]

            depth = spec.intrinsics.fx * baseline / disp

            tensordict["obs", key, "depth"] = torch.unflatten(
                depth.unsqueeze(0), dim=0, sizes=leading_dims
            )

        return tensordict


def load_engine(engine_path: str | os.PathLike) -> trt.ICudaEngine:
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_bindings(engine, device) -> dict[str, Binding]:
    bindings = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            is_input = False
        else:
            is_input = True
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = tuple(engine.get_tensor_shape(name))
        if -1 in shape:
            raise ValueError(f"Dynamic shape for tensor '{name}': {shape}")
        data = torch.empty(size=shape, dtype=torch.float32, device=device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    return bindings


def run_inference(engine, context, bindings_dict):
    binding_addrs = [
        bindings_dict[engine.get_tensor_name(i)].ptr
        for i in range(engine.num_io_tensors)
    ]
    context.execute_v2(binding_addrs)
