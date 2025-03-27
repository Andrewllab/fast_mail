from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from tensordict.nn import TensorDictModule, TensorDictSequential

from environments.specs import DataSpecs

KeyType = str | tuple[str, ...]

log = logging.getLogger(__name__)


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


def create_tdmodules(transform: Transform) -> list[TensorDictModule]:
    """Given a Transform, wrap it in one TensorDictModule for each mapping
    provided in its key_mappings property.
    """
    log.debug(f"Wrapping {transform.__class__} with TensorDictModule")
    return [
        TensorDictModule(
            transform, in_keys=key_mapping.in_keys, out_keys=key_mapping.out_keys
        )
        for key_mapping in transform.key_mappings
    ]


def wrap_with_tdmodule(cls: type[Transform]) -> type[Transform]:
    """This class decorator wraps a Transform class with one or more
    TensorDictModules after it is instantiated.
    """
    # BALAZS: maybe switch to metaclass

    class TDModuleMeta(cls.__class__):  # Dynamically create a new metaclass
        def __call__(self, *args, **kwargs) -> list[TensorDictModule]:
            transform = super().__call__(*args, **kwargs)
            return create_tdmodules(transform)

    # Set the new metaclass for this class
    cls.__class__ = TDModuleMeta
    return cls


def init_transform_sequence(
    transform_partials: list[Callable[[DataSpecs], Transform]], specs: DataSpecs
) -> tuple[TensorDictSequential, DataSpecs]:

    transform_modules: list[TensorDictModule] = []
    for transform_partial in transform_partials:
        assert isinstance(transform_partial, functools.partial)
        transform = transform_partial(specs)
        assert isinstance(transform, Transform)
        specs = transform.specs
        td_modules = create_tdmodules(transform)
        transform_modules.extend(td_modules)

    return TensorDictSequential(*transform_modules), specs
