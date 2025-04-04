import os
from collections import namedtuple

import numpy as np
import tensorrt as trt
import torch
from torch import Tensor

from environments.specs import DataSpecs, StereoCameraSpec
from transforms.base_transform import KeyMapping, Transform
from utils.FoundationStereo.utils import InputPadder

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))


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
            if isinstance(spec, StereoCameraSpec)
        }

        self._padders = {
            key: InputPadder(spec.shape[-2:], divis_by=32, force_square=False)
            for key, spec in self._input_specs.items()
        }

        # Load tensorRT engine
        self.device = torch.device("cuda")  # TODO: avoid hardcoding this
        self.context = self.engine.create_execution_context()
        self.engine_path = engine_path
        self.engine = load_engine(self.engine_path)
        self.bindings = allocate_bindings(self.engine, self.device)

        self._output_specs = ...

        self._key_iter = iter(self._input_specs)
        self._key = next(self._key_iter)

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("obs", key, "left"), ("obs", key, "left")],
                out_keys=[("obs", f"{key}_depth")],
            )
            for key in self._input_specs
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, left: Tensor, right: Tensor) -> Tensor:

        spec = self._input_specs[self.key]

        assert (
            left.shape == right.shape
        ), "Left and right images must have the same shape"

        # flatten batch dimensions
        if spec.type == "ir":
            # expand intensity images to 3 channels (in torch order)
            leading_dims = left.shape[:-2]
            left = left.flatten(end_dim=-3)  # inclusive
            right = right.flatten(end_dim=-3)
        elif spec.type == "rgb":
            leading_dims = left.shape[:-3]
            left = left.flatten(end_dim=-4)  # inclusive
            right = right.flatten(end_dim=-4)

        # expand intensity images to 3 channels (in torch order)
        if spec.type == "ir":
            left = left.unsqueeze(-3).expand(-1, 3, -1, -1)
            right = right.unsqueeze(-3).expand(-1, 3, -1, -1)

        if spec.type == "rgb" and spec.channel_order == "HWC":
            left = left.permute(0, 3, 1, 2)
            right = right.permute(0, 3, 1, 2)

        # as of this point, all images are in the shape (B, 3, H, W)
        assert left.shape[-3] == right.shape[-3] == 3, "RGB images must have 3 channels"
        _, C, H, W = left.shape

        # pad images dimensions to be divisible by 32
        padder = self._padders[self.key]
        left, right = padder.pad(left, right)

        # foundation stereo
        with torch.amp.autocast("cuda", enabled=True):
            run_inference(self.engine, self.context, self.bindings)

        output_binding = [
            name
            for name in self.bindings
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ][0]
        output = self.bindings[output_binding].data
        disp = output.reshape(self.bindings[output_binding].shape)

        disp = padder.unpad(disp.float())
        disp = disp.data.reshape(H, W)

        if self.remove_invisible:
            yy, xx = torch.meshgrid(
                torch.arange(disp.shape[0]), torch.arange(disp.shape[1]), indexing="ij"
            )
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf

        try:
            self.key = next(self._key_iter)
        except StopIteration:
            self._key_iter = iter(self._input_specs)
            self.key = next(self._key_iter)

        return disp


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
