from __future__ import annotations

import functools
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from omegaconf import DictConfig, ListConfig, OmegaConf
from tensordict.nn import TensorDictModule, TensorDictSequential

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
    in_keys: list[KeyType]
    out_keys: list[KeyType]


class Transform(ABC):
    """Base class for all transforms."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # a transform may also inherit from torch.nn.Module, and therefore have a forward method
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @property
    @abstractmethod
    def key_mappings(self) -> list[KeyMapping]:
        pass

    @property
    @abstractmethod
    def specs(self) -> DataSpecs:
        pass


TransformPartial = Callable[[DataSpecs], Transform]
TransformPartialsDict = Mapping[str, TransformPartial]


def _create_tdmodules(transform: Transform) -> list[TensorDictModule]:
    """Given a Transform, wrap it in one TensorDictModule for each mapping
    provided in its key_mappings property.
    """
    td_modules = [
        TensorDictModule(
            transform, in_keys=key_mapping.in_keys, out_keys=key_mapping.out_keys
        )
        for key_mapping in transform.key_mappings
    ]
    log.debug(
        f"Wrapped <{type(transform).__name__}> with {len(td_modules)} TensorDictModule(s)"
    )
    return td_modules


def _item_to_sort_key(item: tuple[str, Any]) -> float:
    """Extracts a float from the first part of a key in a dictionary item."""
    key, _ = item
    # get the part before the first "_"
    num = key.split("_")[0]
    # convert e.g. 1-1 or 1,1 to 1.1, which can be converted to a float
    # periods are not allowed in keys
    num = num.replace("-", ".").replace(",", ".")
    try:
        return float(num)
    except ValueError:
        raise ValueError(
            f"All transform keys must begin with a number separated by an underscore. Got {key}"
        )


def _init_transform(
    transform: Callable[[DataSpecs], Transform], specs: DataSpecs, wrap: bool = True
) -> tuple[Callable, DataSpecs] | tuple[list[Callable], DataSpecs]:
    assert isinstance(transform, functools.partial)
    log.debug(f"Instantiating transform: <{transform.func.__name__}>")

    # instantiate the transform
    transform_instance = transform(specs)
    assert isinstance(transform_instance, Transform)
    # update the specs
    specs = transform_instance.specs

    if not wrap:
        return [transform_instance], specs

    # wrap the transform in TensorDictModule(s)
    transform_modules = _create_tdmodules(transform_instance)
    if len(transform_modules) == 1:
        return transform_modules[0], specs
    return TensorDictSequential(*transform_modules), specs


def init_transforms(
    transforms: TransformPartialsDict | TransformPartial | None,
    specs: DataSpecs,
    wrap: bool = True,
) -> tuple[Callable, DataSpecs] | tuple[list[Callable], DataSpecs]:
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

    transform_instances = []
    i = 1
    for key, partial in transforms.items():

        name = key.split("_", maxsplit=1)[1]
        log.debug(f"Instantiating transform #{i} '{name}': <{partial.func.__name__}>")
        i += 1

        # instantiate the transform
        transform = partial(specs)
        assert isinstance(transform, Transform)
        # update the specs
        specs = transform.specs
        transform_instances.append(transform)

    if not wrap:
        return transform_instances, specs

    transform_modules = []
    for transform in transform_instances:
        # wrap the transform in TensorDictModule(s)
        transform_modules.extend(_create_tdmodules(transform))

    return TensorDictSequential(*transform_modules), specs


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

    cfg = OmegaConf.load(path)
    if not isinstance(cfg, ListConfig):
        raise ValueError(f"Expected a ListConfig, got {type(cfg)}")
    return cfg
