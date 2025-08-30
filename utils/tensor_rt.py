import os
from collections import defaultdict
from typing import Literal

import numpy as np
import tensorrt as trt
import torch

numpy_to_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def load_engine(
    engine_path: str | os.PathLike,
    log_level: Literal["WARNING", "INFO", "VERBOSE"] = "INFO",
) -> tuple[trt.ICudaEngine, trt.IExecutionContext]:
    with open(engine_path, "rb") as f:
        engine_data = f.read()

    logger = trt.Logger(getattr(trt.Logger, log_level))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    return engine, context


def get_metadata(engine: trt.ICudaEngine):
    metadata = defaultdict(dict)

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        dtype = numpy_to_torch_dtype_dict[dtype]
        # the returned type is trt.Dims, so we convert to tuple
        shape = tuple(engine.get_tensor_shape(name))

        if (mode := engine.get_tensor_mode(name)) == trt.TensorIOMode.INPUT:
            mode = "in"
        else:
            assert mode == trt.TensorIOMode.OUTPUT
            mode = "out"

        metadata[mode][name] = {"dtype": dtype, "shape": shape}

    return metadata


def run_inference(engine: trt.ICudaEngine, context: trt.IExecutionContext, **inputs):
    arg_addrs = []
    outputs = {}
    device = list(inputs.values())[0].device
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        np_dtype = trt.nptype(engine.get_tensor_dtype(name))
        dtype = numpy_to_torch_dtype_dict[np_dtype]
        shape = tuple(engine.get_tensor_shape(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input = inputs[name]
            if input.shape != shape:
                raise ValueError(
                    f"Input shape mismatch for {name}: {input.shape} vs {shape}"
                )
            if input.dtype != dtype:
                raise ValueError(
                    f"Input dtype mismatch for {name}: {input.dtype} vs {dtype}"
                )
            if (
                input.device.type == "cpu"
                and engine.get_tensor_location(name) != trt.TensorLocation.HOST
            ):
                # trt.TensorLocation is either HOST or DEVICE, but we don't know which device it needs to be on
                raise ValueError(
                    f"Input {name} is on CPU, but engine expects it on GPU"
                )

            input = input.contiguous()
            arg_addrs.append(input.data_ptr())
        else:
            assert engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
            output = torch.empty(shape, dtype=dtype, device=device)
            outputs[name] = output
            arg_addrs.append(output.data_ptr())

    context.execute_v2(arg_addrs)

    if len(outputs) == 1:
        return next(iter(outputs.values()))
    return outputs
