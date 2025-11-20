from __future__ import annotations

import functools
import logging
import os
import pickle
import re
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
from omegaconf import ListConfig, OmegaConf
from tensordict import TensorDict

from environments.specs import DataSpecs

__all__ = [
    "KeyMapping",
    "Transform",
    "TransformPartial",
    "TransformPartialsDict",
    "init_transforms",
]

log = logging.getLogger(__name__)


KeyType = str | tuple[str, ...]


@dataclass
class KeyMapping:
    in_keys: KeyType | list[KeyType]
    out_keys: KeyType | list[KeyType]
    args: tuple[Any, ...] = ()


class TransformConstraint(Enum):
    TRAJECTORY_ONLY = auto()
    GPU_ONLY = auto()


class TransformModuleMeta(ABCMeta):
    """Metaclass that allows for a class to multiple inherit from Transform and
    torch.nn.Module. This metaclass prevents Transform.__call__ from shadowing
    torch.nn.Module.__call__.
    """

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        # Check if nn.Module is in the MRO
        if any(issubclass(base, nn.Module) for base in cls.__mro__[1:]):
            # Override __call__, with the one from nn.Module
            cls.__call__ = nn.Module.__call__

            # Override train and eval, with the ones from nn.Module
            if "eval" not in namespace:
                cls.eval = nn.Module.eval

            if "train" not in namespace:
                cls.train = nn.Module.train
        return cls


class Transform(ABC, metaclass=TransformModuleMeta):
    """Base class for all transforms."""

    constraints: list[TransformConstraint] = []
    training: bool = True

    @property
    @abstractmethod
    def specs(self) -> DataSpecs:
        pass

    @property
    def key_mappings(self) -> list[KeyMapping]:
        raise NotImplementedError

    @property
    def mapping_idx(self) -> int:
        """Available at runtime for _call_one method of child classes. Provides
        the index of the current key mapping, allowing child classes to modify
        their behavior based on the spec of the input field.
        """
        return self._mapping_idx

    def train(self, mode: bool = True):
        self.training = mode

    def eval(self):
        self.train(mode=False)

    def __call__(self, tensordict: TensorDict) -> TensorDict:

        try:
            key_mappings = self.key_mappings
        except NotImplementedError as e:
            raise NotImplementedError(
                "A transform must either define a key_mappings property or implement __call__."
            ) from e

        for idx, key_mapping in enumerate(key_mappings):
            self._mapping_idx = idx

            in_keys = key_mapping.in_keys
            if not isinstance(in_keys, list):
                in_keys = [in_keys]

            # we get the inputs with a default value of None, allowing
            # support for missing keys
            inputs = (tensordict.get(key, None) for key in in_keys)
            outputs = self._call_one(*inputs, *key_mapping.args)

            out_keys = key_mapping.out_keys
            if out_keys == "_":
                # if out_keys is "_", no writeback is performed
                continue
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            if not isinstance(out_keys, list):
                out_keys = [out_keys]
            for out_key, output in zip(out_keys, outputs):
                tensordict.set(out_key, output)

        return tensordict

    def _call_one(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # a transform may also inherit from torch.nn.Module, and therefore have a
    # forward method in this case, the metaclass ensures that the __call__
    # method of torch.nn.Module is not shadowed, and forward takes on the role
    # of __call__
    forward = __call__

    def call_trajectory(
        self, tensordict: TensorDict
    ) -> TensorDict | Sequence[TensorDict]:
        """Transform a tensordict representing a complete trajectory (rather
        than a batch of samples). This is used during preprocessing, as it
        gives the transform more freedom to modify the data.

        By default, call_trajectory is the same as __call__/forward, but can
        be overridden in child classes.
        """
        return self(tensordict)


class ReversibleTransform(Transform):
    @property
    def reverse_key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=key_mapping.out_keys,
                out_keys=key_mapping.in_keys,
                args=key_mapping.args,
            )
            for key_mapping in self.key_mappings
        ]

    def reverse(self, tensordict: TensorDict) -> TensorDict:

        try:
            key_mappings = self.reverse_key_mappings
        except NotImplementedError as e:
            raise NotImplementedError(
                "A transform must either define a key_mappings property or implement __call__."
            ) from e

        for idx, key_mapping in enumerate(key_mappings):
            self._mapping_idx = idx

            in_keys = key_mapping.in_keys
            if not isinstance(in_keys, list):
                in_keys = [in_keys]

            # we get the inputs with a default value of None, allowing
            # support for missing keys
            inputs = (tensordict.get(key, None) for key in in_keys)
            outputs = self._reverse_one(*inputs, *key_mapping.args)

            out_keys = key_mapping.out_keys
            if out_keys == "_":
                # if out_keys is "_", no writeback is performed
                continue
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            if not isinstance(out_keys, list):
                out_keys = [out_keys]
            for out_key, output in zip(out_keys, outputs):
                tensordict.set(out_key, output)

        return tensordict

    def _reverse_one(self, *args: Any, **kwargs: Any) -> Any:
        """Reverses the transformation applied by this transform for a single key mapping."""
        raise NotImplementedError(
            "ReversibleTransform must implement reverse_one method to reverse the transformation for a single key mapping."
        )


class NormalizingTransform(ReversibleTransform):
    """A normalizing transform is one that executes in 3 different contexts.
    It implements call_trajectory to collect dataset statistics during
    preprocessing. Its __call__ (or _call_one) method normalizes the data, and
    its reverse (or _reverse_one) method un-normalizes the data.

    A normalization that doesn't require dataset statistics can be implemented
    as a simple ReversibleTransform, which doesn't need to save state or run
    during preprocessing.
    """

    pass


class Compose(ReversibleTransform):
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (``Transform`` objects): transforms to compose.

    Example:
        >>> Compose(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> )

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Modified from: https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py#L60
    """

    # TODO: add __new__ method that either creates a Compose or Sequential based on
    # whether any of the transforms is an nn.Module

    def __init__(
        self,
        *transforms: Transform | Mapping[str, Transform],
        specs: DataSpecs | None = None,
    ):
        self._transforms: dict[str, Transform] = {}  # similar to nn.Module._modules

        if len(transforms) == 1 and isinstance(transforms[0], Mapping):
            for key, transform in transforms[0].items():
                self._transforms[key] = transform
        else:
            for idx, transform in enumerate(transforms):
                assert isinstance(transform, Transform)
                key = str(idx)
                self._transforms[key] = transform

        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        if self._transforms:
            return list(self._transforms.values())[
                -1
            ].specs  # specs of the last transform
        else:
            if self._specs is None:
                raise RuntimeError(
                    "This Compose transform is empty, but no specs were provided. Please provide them on initialization."
                )
            return self._specs

    def train(self, mode: bool = True):
        for t in self._transforms.values():
            t.train(mode=mode)
        self.training = mode

    def __getitem__(self, idx: int | str | slice) -> Transform | list[Transform]:
        if isinstance(idx, str):
            return self._transforms[idx]
        elif isinstance(idx, (int, slice)):
            return list(self._transforms.values())[idx]
        else:
            raise TypeError(f"Expected idx to be an int, str or slice, got {type(idx)}")

    def __setitem__(
        self, idx: int | str | slice, transform: Transform | Sequence[Transform]
    ) -> None:
        if isinstance(idx, str):
            assert isinstance(transform, Transform)
            self._transforms[idx] = transform
        elif isinstance(idx, (int, slice)):
            key = list(self._transforms.keys())[idx]
            if isinstance(key, list):
                assert isinstance(transform, Sequence)
                for k, t in zip(key, transform):
                    self._transforms[k] = t
            else:
                assert isinstance(transform, Transform)
                self._transforms[key] = transform
        else:
            raise TypeError(f"Expected idx to be an int, str or slice, got {type(idx)}")

    def __len__(self) -> int:
        return len(self._transforms)

    def __iter__(self) -> Iterator[Transform]:
        return iter(self._transforms.values())

    def keys(self) -> Iterator[str]:
        return self._transforms.keys()

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        for t in self._transforms.values():
            tensordict = t(tensordict)
        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        # apply transforms in reverse order
        for t in reversed(self._transforms.values()):
            if not isinstance(t, ReversibleTransform):
                raise TypeError(
                    f"Cannot reverse {t} as it is not a ReversibleTransform."
                )
            tensordict = t.reverse(tensordict)
        return tensordict

    def __add__(self, other: Compose) -> Compose:
        if not isinstance(other, Compose):
            raise TypeError(
                f"Can only add Compose to another Compose (got {type(other)})"
            )
        # concatenate the transforms in self and other
        transforms = list(self) + list(other)
        cls = (
            Sequential if any(isinstance(t, nn.Module) for t in transforms) else Compose
        )
        return cls(*transforms, specs=other.specs)

    def __repr__(self) -> str:
        if not self._transforms:
            return f"{self.__class__.__name__}()"

        parts = [self.__class__.__name__ + "("]
        for name, t in self._transforms.items():
            parts.append(f"    {name}: {t}")
        parts.append(")")
        return "\n".join(parts)


class Sequential(nn.Module, Compose):
    """This class slightly modifies the torch.nn.Sequential class to allow for
    inputs that are not nn.Module.

    Modified from: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/container.py#L54
    """

    def __init__(
        self,
        *transforms: Transform | Mapping[str, Transform],
        specs: DataSpecs | None = None,
    ):
        super().__init__()  # nn.Module.__init__

        # similar to nn.Module._modules, but not only for nn.Module instances
        self._transforms: dict[str, Transform] = {}

        if len(transforms) == 1 and isinstance(transforms[0], Mapping):
            for key, transform in transforms[0].items():
                self._transforms[key] = transform
                setattr(self, key, transform)  # handles Module registration
        else:
            for idx, transform in enumerate(transforms):
                assert isinstance(transform, Transform)
                key = str(idx)
                self._transforms[key] = transform
                setattr(self, key, transform)  # handles Module registration

        self._specs = specs

    def __dir__(self):
        keys = super().__dir__()
        # filter out any keys that are just indices
        keys = [key for key in keys if not key.isdigit()]
        return keys

    # nn.Module's __repr__ shadows Compose's __repr__, so we need to explicitly assign it
    __repr__ = Compose.__repr__

    def train(self, mode: bool = True):
        # First call nn.Module's train to set self.training
        super().train(mode=mode)

        # Set training mode for each transform
        for t in self._transforms.values():
            t.train(mode=mode)

    def forward(self, input: TensorDict) -> TensorDict:
        for module in self:
            input = module(input)
        return input


DeviceType = torch.device | str | int


class GpuExecutionWrapper(Transform):
    def __init__(
        self,
        transform: Transform,
        device: DeviceType | None = None,
        batch_size: int | None = None,
    ) -> None:
        if (
            TransformConstraint.TRAJECTORY_ONLY in transform.constraints
            and batch_size is not None
        ):
            raise ValueError(
                f"{transform.__class__.__name__} does not support execution in chunks."
            )

        device = device or getattr(transform, "device", None) or "cuda"
        batch_size = batch_size or getattr(transform, "batch_size", None)

        self._transform = transform
        self.device = device
        self.batch_size = batch_size

    @property
    def __class__(self):
        # the wrapper pretends to be the same type as the wrapped transform
        # so that isinstance(transform, nn.Module) is still True if the
        # wrapped transform is an nn.Module.
        return type(self._transform)

    @property
    def specs(self) -> DataSpecs:
        return self._transform.specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if self.batch_size is not None:
            return self._call_in_chunks(tensordict)

        # if no batch size is given, just move the entire tensordict to the GPU
        return self._transform(tensordict.to(self.device)).cpu()

    def call_trajectory(
        self, tensordict: TensorDict
    ) -> TensorDict | Sequence[TensorDict]:
        if self.batch_size is not None:
            # this method uses __call__ instead of call_trajectory, and we
            # to assume that this has the same effect as call_trajectory,
            # because we don't know how to handle multiple chunks returned
            # for each input chunk.
            return self._call_in_chunks(tensordict)

        # if no batch size is given, just move the entire trajectory to the GPU
        transformed = self._transform.call_trajectory(tensordict.to(self.device))

        if isinstance(transformed, Sequence):
            return [t.cpu() for t in transformed]
        else:
            return transformed.cpu()

    def _call_in_chunks(self, tensordict: TensorDict) -> TensorDict:

        # Split up a trajectory into chunks that fit into GPU memory,
        # apply the transform to each chunk, and concatenate the results
        to_chunk, rest = tensordict.split_keys(["obs", "action", "ref_action"])
        to_chunk.auto_batch_size_(batch_dims=1)
        assert to_chunk.ndim == 1

        log.debug(
            f"Applying transform {self._transform.__class__.__name__} on trajectory of length {to_chunk.shape[0]} in chunks of size {self.batch_size} on {self.device} device..."
        )

        # for memory reasons, we don't want to store each transformed chunk
        # in memory and concatenate at the end
        for i in range(0, to_chunk.shape[0], self.batch_size):
            chunk = to_chunk[i : i + self.batch_size]

            # add the chunked fields back to the non-chunked fields
            # we assume that the transform does not modify the other fields
            # or the tensordict itself
            input_td = rest.clone()
            input_td.update(chunk)

            # We don't call transform.call_trajectory here because we are
            # not operating on an entire trajectory
            transformed = self._transform(input_td.to(self.device)).cpu()

            # Write back immediately to avoid accumulating large tensors in memory
            # `update` doesn't work because transformed only contains a chunk
            # `update_at_` doesn't work because the tensordicts may not have
            # matching structure if the transform added any new fields
            to_chunk[i : i + self.batch_size] = transformed.select("obs", "action")

            # Free up cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tensordict.update(to_chunk)

        return tensordict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._transform)})"


TransformPartial = Callable[[DataSpecs], Transform]
TransformPartialsDict = Mapping[str, TransformPartial]


# Regex pattern explanation:
# t                 — literal 't'
# (\d+)             — first group of digits
# (?:[-,](\d+))?    — optional dash or comma followed by second group of digits
# _                 — literal underscore
# (.+)              — everything after the underscore
KEY_PATTERN = re.compile(r"t(\d+)(?:[-,](\d+))?_(.+)")


def _item_to_sort_key(item: tuple[str, Any]) -> float:
    """Extracts a float from the first part of a key in a dictionary item."""
    key, _ = item

    match = KEY_PATTERN.fullmatch(key)
    if match:
        first_number = match.group(1)
        second_number = match.group(2)  # may be None
        ordinal = float(f"{first_number}.{second_number if second_number else 0}")
        return ordinal
    else:
        raise ValueError(
            f"All transform keys must match the following pattern: t##_name or t##-##_name or t##,##_name. Got {key}"
        )


def _parse_key(key: str) -> tuple[str, str]:
    match = KEY_PATTERN.fullmatch(key)
    if match:
        first_number = match.group(1)
        second_number = match.group(2)  # may be None
        name = match.group(3)
        ordinal = first_number
        if second_number is not None:
            ordinal += "." + second_number
        return ordinal, name
    else:
        raise ValueError(
            f"All transform keys must match the following pattern: t##_name or t##-##_name or t##,##_name. Got {key}"
        )


def init_transforms(
    transforms: TransformPartialsDict | None,
    specs: DataSpecs,
) -> tuple[Compose, DataSpecs]:
    """Instantiates a sequence of transforms from a dictionary of transform partials,
    while propagating the specs through the sequence.

    :param transforms: A mapping containing transform partials, where the first part of the key
    (before the "_") acts as the sort key.
    :return: The instantiated transforms wrapped in a TensorDictSequential, and the final specs.
    """
    if transforms is None:
        return Compose(specs=specs), specs

    # filter out any values that are not partials
    transforms = {
        k: v for k, v in transforms.items() if isinstance(v, functools.partial)
    }

    # sort dictionary of transforms by the first part of the key, which should be a number
    transforms = dict(sorted(transforms.items(), key=_item_to_sort_key))

    # by passing the transforms as an OrderedDict to torch.nn.Sequential, we
    # get the benefit of sane naming of transforms
    transform_instances = OrderedDict()
    for key, partial in transforms.items():

        assert isinstance(partial, functools.partial)
        ordinal, name = _parse_key(key)
        log.info(
            f"Instantiating transform #{ordinal} '{name}': <{partial.func.__name__}>"
        )

        # instantiate the transform
        transform = partial(specs)
        assert isinstance(transform, Transform)
        # update the specs
        specs = transform.specs

        key = f"{ordinal}_{name}"
        assert key not in transform_instances
        transform_instances[key] = transform

    cls = (
        Sequential
        if any(isinstance(t, nn.Module) for t in transform_instances.values())
        else Compose
    )

    return cls(transform_instances, specs=specs), specs


def get_transforms_config(transform_partials: TransformPartialsDict) -> ListConfig:
    """Converts a dictionary of transform partials to a minimal config. The
    returned config is a ListConfig, where metadata such as transform names
    or non-partial items have been filtered out.

    :param transform_partials: A mapping containing transform_partials, where
        the first part of the key (before the "_") acts as the sort key
    """

    # filter out any values that are not partials
    transform_partials = {
        k: v for k, v in transform_partials.items() if isinstance(v, functools.partial)
    }

    # sort dictionary of transform_partials by the first part of the key, which should be a number
    transform_partials = dict(sorted(transform_partials.items(), key=_item_to_sort_key))

    cfg = []
    for partial in transform_partials.values():
        cfg.append(
            {
                "name": partial.func.__name__,
                "args": partial.args,
                "kwargs": partial.keywords,
            }
        )

    # convert to omegaconf ListConfig
    cfg = OmegaConf.create(cfg)
    assert isinstance(cfg, ListConfig)
    return cfg


def save_transforms_config(
    transform_partials: TransformPartialsDict, path: os.PathLike
) -> None:
    cfg = get_transforms_config(transform_partials)
    with open(path, "w") as f:
        OmegaConf.save(cfg, f)


def load_transforms_config(path: os.PathLike) -> ListConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transforms config file {path} does not exist")

    cfg = OmegaConf.load(str(path))
    if not isinstance(cfg, ListConfig):
        raise ValueError(f"Expected a ListConfig, got {type(cfg)}")
    return cfg


def save_transforms(transforms: Transform, path: os.PathLike) -> None:
    """Saves the transforms to a file."""
    with open(path, "wb") as f:
        pickle.dump(transforms, f)


def load_transforms(path: os.PathLike) -> Transform:
    with open(path, "rb") as f:
        transforms = pickle.load(f)

    return transforms
