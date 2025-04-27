from __future__ import annotations

from pathlib import Path

import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict, is_leaf_nontensor
from torch_geometric.data import Batch, Data

from environments.specs import CameraSpec, DataSpecs, RGBStream
from utils.pyg import update_batch_metadata


def compress_rgb_images(tensordict: TensorDict, specs: DataSpecs) -> TensorDict:
    """Convert any RGB images in the tensordict from float32 to uint8 to reduce
    memory consumption.
    """
    for key, spec in specs.obs.items():
        if not isinstance(spec, CameraSpec):
            continue

        for name, stream in spec.streams.items():
            if isinstance(stream, RGBStream):

                image = tensordict["obs", key, name]
                if image.dtype != torch.uint8:
                    image = image.mul(255).clamp(0, 255).to(torch.uint8)
                    tensordict["obs", key, name] = image

                # if the stream is RGB, we need to convert it to uint8
                tensordict["obs", key, name] = tensordict["obs", key, name].to(
                    dtype=torch.uint8
                )

    return tensordict


def reduce_pyg_data(tensordict: TensorDict) -> TensorDict:
    """Reduce any pyg objects in the tensordict to prepare for saving.
    - NonTensorStacks of Data objects are converted to Batch objects, because
        tensordict.memmap() is very inefficient for saving NonTensorStacks.
    - Batch objects get their metadata updated, so that the batch elements can
        be reconstricted after loading.
    """
    for key, value in tensordict.items(
        include_nested=True, leaves_only=True, is_leaf=is_leaf_nontensor
    ):
        if (
            isinstance(value, NonTensorStack)
            and isinstance(value.tolist()[0], Data)
            and not isinstance(value.tolist()[0], Batch)
        ):
            data = Batch.from_data_list(value.tolist())
            tensordict[key] = data
        elif isinstance(value, NonTensorData) and isinstance(value.data, Data):
            tensordict[key] = update_batch_metadata(value.data)

    return tensordict


def save_tensordict(
    tensordict: TensorDict, file: Path, specs: DataSpecs, backend: str = "memmap"
):
    """Save a tensordict to a file. If the file is a directory, it is
    assumed to be a memory-mapped tensordict and the tensordict is saved
    to the directory.

    Args:
        tensordict (TensorDict): The tensordict to save.
        file (Path): The file to save to.
    """
    # remove batch dimensions so we can add fields of any shape
    tensordict.auto_batch_size_(batch_dims=0)

    tensordict = compress_rgb_images(tensordict, specs)

    # convert Data objects to Batch objects
    tensordict = reduce_pyg_data(tensordict)

    if backend == "memmap":
        tensordict.memmap(str(file), num_threads=8)
    else:
        raise NotImplementedError(
            f"Backend {backend} not implemented. Only memmap is supported."
        )


def unreduce_pyg_data(tensordict: TensorDict) -> TensorDict:
    """Reverse any reduction done before saving to restore the original pyg
    objects. This is the inverse of reduce_pyg_data.
    - Batch objects are converted to NonTensorStacks of Data objects, so that
        we can index along the leading (time) dimension.
    """
    for key, value in tensordict.items(
        include_nested=True, leaves_only=True, is_leaf=is_leaf_nontensor
    ):
        if isinstance(value, NonTensorData) and isinstance(value.data, Batch):
            # wrap each point cloud in a NonTensorData object
            datas = value.data.to_data_list()
            # create a NonTensorStack object from the list of NonTensorData objects
            nt_stack = NonTensorStack(*datas)
            # data = data_to_tensordict(data)
            tensordict[key] = nt_stack

    return tensordict


def load_tensordict(
    file: Path,
    start: int | None = None,
    stop: int | None = None,
    backend: str = "memmap",
) -> TensorDict:
    """Load a tensordict from a file. If the file is a directory, it is
    assumed to be a memory-mapped tensordict and the start and stop
    indices are used to slice it.

    Args:
        file (Path): The file to load.
        start (int | None): The start index to slice the tensordict.
        stop (int | None): The stop index to slice the tensordict.

    Returns:
        TensorDict: The loaded tensordict.
    """
    if backend == "memmap":
        assert file.is_dir()
        # this is a memory-mapped tensordict
        trajectory = TensorDict.load_memmap(file, non_blocking=True)
    else:
        raise NotImplementedError(
            f"Backend {backend} not implemented. Only memmap is supported."
        )

    # convert DataBatch objects back into Data objects
    trajectory = unreduce_pyg_data(trajectory)

    # we don't need to decompress rgb images here, since uint8 is also a valid
    # way to store images

    # add back the batch dimension so we can index along the leading (time) dimension
    trajectory.auto_batch_size_(batch_dims=1)

    if start is not None or stop is not None:
        # slice the tensordict
        trajectory = trajectory[start:stop]

    return trajectory
