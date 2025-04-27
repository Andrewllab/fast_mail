from __future__ import annotations

import functools
import logging
import os
import re
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

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

    def __call__(self, tensordict: TensorDict) -> Any:

        try:
            for idx, key_mapping in enumerate(self.key_mappings):
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

        except NotImplementedError as e:
            raise NotImplementedError(
                "A transform must either define a key_mappings property or implement __call__."
            ) from e

    def _call_one(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # a transform may also inherit from torch.nn.Module, and therefore have a
    # forward method in this case, the metaclass ensures that the __call__
    # method of torch.nn.Module is not shadowed, and forward takes on the role
    # of __call__
    forward = __call__

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        """Transform a tensordict representing a complete trajectory (rather
        than a batch of samples). By default, this is the same as
        __call__/forward, but can be overridden in child classes.
        """
        return self(tensordict)


class Compose:
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

    def __init__(self, *transforms: Callable | Mapping[str, Callable]):
        self._transforms: dict[str, Callable] = {}  # similar to nn.Module._modules

        if len(transforms) == 1 and isinstance(transforms[0], Mapping):
            for key, module in transforms[0].items():
                self._transforms[key] = module
        else:
            for idx, module in enumerate(transforms):
                self._transforms[str(idx)] = module

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        for t in self._transforms.values():
            tensordict = t(tensordict)
        return tensordict

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self._transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class Sequential(nn.Module):
    """This class slightly modifies the torch.nn.Sequential class to allow for
    inputs that are not nn.Module.

    Modified from: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/container.py#L54
    """

    def __init__(self, *transforms: Callable | Mapping[str, Callable]):
        super().__init__()

        # similar to nn.Module._modules, but not only for nn.Module instances
        self._transforms: dict[str, Callable] = {}

        if len(transforms) == 1 and isinstance(transforms[0], Mapping):
            for key, transform in transforms[0].items():
                self._transforms[key] = transform
                setattr(self, key, transform)  # handles Module registration
        else:
            for idx, transform in enumerate(transforms):
                key = str(idx)
                self._transforms[key] = transform
                setattr(self, key, transform)  # handles Module registration

    def __getitem__(self, idx: slice | int) -> Sequential | Callable:
        if isinstance(idx, slice):
            return self.__class__(dict(list(self._transforms.items())[idx]))
        else:
            return list(self._transforms.values())[idx]

    def __setitem__(self, idx: int, transform: Callable) -> None:
        key: str = list(self._transforms.keys())[idx]
        self._transforms[key] = transform
        return setattr(self, key, transform)

    def __len__(self) -> int:
        return len(self._transforms)

    def __dir__(self):
        keys = super().__dir__()
        # filter out any keys that are just indices
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self) -> Iterator[Callable]:
        return iter(self._transforms.values())

    def forward(self, input):
        for module in self:
            input = module(input)
        return input

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self._transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


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


def _init_transform(
    transform: Callable[[DataSpecs], Transform], specs: DataSpecs, wrap: bool = True
) -> tuple[Transform, DataSpecs] | tuple[list[Transform], DataSpecs]:
    assert isinstance(transform, functools.partial)
    log.debug(f"Instantiating transform: <{transform.func.__name__}>")

    # instantiate the transform
    transform_instance = transform(specs)
    assert isinstance(transform_instance, Transform)
    # update the specs
    specs = transform_instance.specs

    if not wrap:
        return [transform_instance], specs

    return transform_instance, specs


def init_transforms(
    transforms: TransformPartialsDict | TransformPartial | None,
    specs: DataSpecs,
    wrap: bool = True,
) -> tuple[Callable, DataSpecs] | tuple[list[Transform], DataSpecs]:
    """Instantiates a sequence of transforms from a dictionary of transform partials,
    while propagating the specs through the sequence.

    :param transforms: A mapping containing transforms, where the first part of the key
    (before the "_") acts as the sort key.
    :return: The instantiated transforms wrapped in a TensorDictSequential, and the final specs.
    """
    if transforms is None:
        return lambda x: x, specs

    if callable(transforms):
        return _init_transform(transforms, specs, wrap=wrap)

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

    if not wrap:
        return list(transform_instances.values()), specs

    cls = (
        Sequential
        if any(isinstance(t, nn.Module) for t in transform_instances.values())
        else Compose
    )

    return cls(transform_instances), specs


def get_transforms_config(transforms: TransformPartialsDict) -> ListConfig:
    """Converts a dictionary of transform partials to a minimal config. The
    returned config is a ListConfig, where metadata such as transform names
    or non-partial items have been filtered out.

    :param transforms: A mapping containing transforms, where the first part of the key
    """

    # filter out any values that are not partials
    transforms = {
        k: v for k, v in transforms.items() if isinstance(v, functools.partial)
    }

    # sort dictionary of transforms by the first part of the key, which should be a number
    transforms = dict(sorted(transforms.items(), key=_item_to_sort_key))

    cfg = []
    for partial in transforms.values():
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
    transforms: TransformPartialsDict, path: os.PathLike
) -> None:
    cfg = get_transforms_config(transforms)
    with open(path, "w") as f:
        OmegaConf.save(cfg, f)


def load_transforms_config(path: os.PathLike) -> ListConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transforms config file {path} does not exist")

    cfg = OmegaConf.load(str(path))
    if not isinstance(cfg, ListConfig):
        raise ValueError(f"Expected a ListConfig, got {type(cfg)}")
    return cfg
