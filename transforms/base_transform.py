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
            # Override __call__ with the one from nn.Module
            cls.__call__ = nn.Module.__call__
        return cls


class Transform(ABC, metaclass=TransformModuleMeta):
    """Base class for all transforms."""

    constraints: list[TransformConstraint] = []

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
            # TODO: ensure that keys are unique, otherwise we silently skip
            # transforms
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

    def __getitem__(self, idx: int | str | slice) -> Transform | list[Transform]:
        if isinstance(idx, str):
            return self._transforms[idx]
        elif isinstance(idx, (int, slice)):
            return list(self._transforms.values())[idx]
        else:
            raise TypeError(f"Expected idx to be an int, str or slice, got {type(idx)}")

    def __len__(self) -> int:
        return len(self._transforms)

    def __iter__(self) -> Iterator[Transform]:
        return iter(self._transforms.values())

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

    def forward(self, input: TensorDict) -> TensorDict:
        for module in self:
            input = module(input)
        return input


TransformPartial = Callable[[DataSpecs], Transform]
TransformPartialsDict = Mapping[str, TransformPartial]


# Regex pattern explanation:
# t                 — literal 't'
# (\d+)             — first group of digits
# (?:[-,](\d+))?    — optional dash or comma followed by second group of digits
# _                 — literal underscore
# (.+)              — everything after the underscore
pattern = re.compile(r"t(\d+)(?:[-,](\d+))?_(.+)")


def _item_to_sort_key(item: tuple[str, Any]) -> float:
    """Extracts a float from the first part of a key in a dictionary item."""
    key, _ = item

    match = pattern.fullmatch(key)
    if match:
        first_number = match.group(1)
        second_number = match.group(2)  # may be None
        ordinal = float(f"{first_number}.{second_number if second_number else 0}")
        return ordinal
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
    i = 1
    for key, partial in transforms.items():

        assert isinstance(partial, functools.partial)
        name = pattern.fullmatch(key).group(3)
        log.debug(f"Instantiating transform #{i} '{name}': <{partial.func.__name__}>")
        i += 1

        # instantiate the transform
        transform = partial(specs)
        assert isinstance(transform, Transform)
        # update the specs
        specs = transform.specs
        transform_instances[name] = transform

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

    # convert to omegaconf DictConfig
    cfg = OmegaConf.create(cfg)
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
