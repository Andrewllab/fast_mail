from __future__ import annotations

import dataclasses
from collections.abc import Mapping, MutableMapping
from operator import getitem
from typing import Any, Callable, Generic, Iterable, Iterator, Sequence, TypeVar

import numpy as np
import torch

from utils.nested import is_torch_nested_tensor

from .types import ArrayLike, ArrayOrMapping, ArrayTree, ArrayType

_T = TypeVar("_T")


def _nested_unbind_list(nt):
    # nt is a torch NestedTensor
    # unbind() returns a tuple of (non-nested) tensors
    return list(nt.unbind())


def _nested_getitem(nt, key):
    """
    NestedTensor-safe equivalent of nt[key].

    - int -> returns a single Tensor
    - slice / sequence of indices / 1D LongTensor / 1D ndarray -> returns NestedTensor
    """
    key = _normalize_nested_key(key)
    parts = _nested_unbind_list(nt)

    # int indexing => return leaf tensor
    if isinstance(key, (int, np.integer)):
        return parts[int(key)]

    # slice => return NestedTensor of sliced parts
    if isinstance(key, slice):
        return torch.nested.nested_tensor(
            parts[key], device=nt.device, layout=nt.layout
        )

    # numpy integer array / list/tuple of ints
    if isinstance(key, np.ndarray):
        if key.dtype == np.bool_:
            key = np.nonzero(key)[0]
        key = key.tolist()

    # torch index tensor
    if torch is not None and isinstance(key, torch.Tensor):
        if key.dtype == torch.bool:
            key = torch.nonzero(key, as_tuple=False).flatten()
        key = key.to(dtype=torch.long).flatten().tolist()

    if isinstance(key, Sequence) and not isinstance(key, (str, bytes)):
        idx = [int(i) for i in key]
        return torch.nested.nested_tensor(
            [parts[i] for i in idx], device=nt.device, layout=nt.layout
        )

    raise TypeError(f"Unsupported index type for NestedTensor: {type(key)!r}")


def _nested_setitem(nt, key, value):
    """
    NestedTensor-safe equivalent of nt[key] = value.

    Since NestedTensor doesn't reliably support in-place setitem,
    we rebuild and return a new NestedTensor.
    """
    key = _normalize_nested_key(key)
    parts = _nested_unbind_list(nt)

    # Normalize key into something Python-list-assignable
    if isinstance(key, np.ndarray):
        if key.dtype == np.bool_:
            key = np.nonzero(key)[0]
        key = key.tolist()

    if torch is not None and isinstance(key, torch.Tensor):
        if key.dtype == torch.bool:
            key = torch.nonzero(key, as_tuple=False).flatten()
        key = key.to(dtype=torch.long).flatten().tolist()

    # int assignment
    if isinstance(key, (int, np.integer)):
        parts[int(key)] = value
        return torch.nested.nested_tensor(parts, device=nt.device, layout=nt.layout)

    # slice assignment
    if isinstance(key, slice):
        # value should be a sequence of tensors of matching length
        parts[key] = value
        return torch.nested.nested_tensor(parts, device=nt.device, layout=nt.layout)

    # sequence assignment
    if isinstance(key, Sequence) and not isinstance(key, (str, bytes)):
        idx = [int(i) for i in key]
        # value can be broadcast (single tensor) or per-index sequence
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != len(idx):
                raise ValueError(
                    f"Length mismatch: {len(value)} values for {len(idx)} indices"
                )
            for i, v in zip(idx, value):
                parts[i] = v
        else:
            for i in idx:
                parts[i] = value
        return torch.nested.nested_tensor(parts, device=nt.device, layout=nt.layout)

    raise TypeError(
        f"Unsupported index type for NestedTensor assignment: {type(key)!r}"
    )


def _normalize_nested_key(key):
    # Treat "..." as "take everything"
    if key is Ellipsis:
        return slice(None)

    # Handle tuple indexing like (..., i), (i, ...), (..., :)
    if isinstance(key, tuple):
        # Remove Ellipsis entries
        non_ellipsis = [k for k in key if k is not Ellipsis]

        # () or (...) => take all
        if len(non_ellipsis) == 0:
            return slice(None)

        # If there is more than one actual index, NestedTensor "outer dim only"
        # can't represent that safely.
        if len(non_ellipsis) > 1:
            raise TypeError(
                f"NestedTensor only supports outer-dimension indexing here; got key={key!r}"
            )

        return non_ellipsis[0]

    return key


class ArrayDict(MutableMapping, Generic[ArrayType]):
    """A batched dictionary of array-like objects.

    ArrayDict is a container for array-like objects (e.g. np.ndarray,
    torch.Tensor, parllel Arrays, etc.) that are stored as key-value pairs,
    where all arrays have leading batch dimensions in common.

    This class is heavily inspired by torch's TensorDict.
    """

    # TODO: consider adding a method to get nested item
    # TODO: consider adding a method to get a subset of keys as a new ArrayDict
    def __init__(
        self,
        items: ArrayOrMapping | Iterable[tuple[str, ArrayOrMapping]] | None = None,
        _run_checks: bool = True,
    ) -> None:
        dict_ = dict(items) if items is not None else {}

        if _run_checks:
            # clean tree to ensure only leaf nodes or ArrayDicts
            # this also shallow copies all the Mapping objects without copying
            # the leaf nodes
            for key, value in dict_.items():
                if isinstance(value, Mapping):
                    dict_[key] = ArrayDict(value)

        self._dict: dict[str, ArrayTree[ArrayType]] = dict_

    def get(self, key: str, default: _T = None) -> ArrayTree[ArrayType] | _T:
        return self._dict.get(key, default)

    def __getitem__(self, key: Any) -> ArrayTree[ArrayType]:
        if isinstance(key, str):
            return self._dict[key]

        try:

            def _index(arr):
                if isinstance(arr, ArrayDict):
                    return arr[key]
                if is_torch_nested_tensor(arr):
                    return _nested_getitem(arr, key)
                return arr[key]

            return ArrayDict(
                ((field, _index(arr)) for field, arr in self._dict.items()),
                _run_checks=False,
            )
        except IndexError as e:
            for field, arr in self._dict.items():
                try:
                    _ = (
                        arr[key]
                        if not is_torch_nested_tensor(arr)
                        else _nested_getitem(arr, key)
                    )
                except IndexError:
                    raise IndexError(
                        f"Index error in field '{field}' for index '{key}'"
                    ) from e
            raise e

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            self._dict[key] = value
            return

        if isinstance(value, Mapping):
            getter = getitem
            fields = self._dict.keys() & value.keys()
        elif dataclasses.is_dataclass(value):
            getter = getattr
            fields = self._dict.keys() & set(dataclasses.fields(value))
        else:
            getter = lambda obj, field: obj
            fields = self._dict.keys()

        for field in fields:
            arr = self._dict[field]
            subvalue = getter(value, field)

            try:
                if isinstance(arr, ArrayDict):
                    arr[key] = subvalue
                elif is_torch_nested_tensor(arr):
                    # rebuild and replace (NestedTensor may not support in-place setitem)
                    self._dict[field] = _nested_setitem(arr, key, subvalue)
                else:
                    arr[key] = subvalue
            except IndexError as e:
                raise IndexError(
                    f"Index error in field '{field}' for index '{key}'"
                ) from e

    def __delitem__(self, __key: str) -> None:
        if isinstance(__key, str):
            del self._dict[__key]
        else:
            raise IndexError(f"Cannot delete index {__key}")

    def __iter__(self) -> Iterator:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return repr(self._dict)

    def __getattr__(self, name: str) -> ArrayAttrDict:
        try:
            return ArrayAttrDict(
                ((field, getattr(arr, name)) for field, arr in self._dict.items()),
                name=name,
                _run_checks=False,
                # _run_checks=True would converted nested ArrayAttrDicts back
                # to ArrayDict
            )
        except AttributeError as e:
            for field, arr in self._dict.items():
                try:
                    _ = getattr(arr, name)
                except IndexError:
                    raise IndexError(
                        f"Attribute error in field '{field}' for attribute '{name}'"
                    ) from e
            raise e

    def __getstate__(self) -> dict[str, Any]:
        # define getstate and setstate explicitly so that pickle does not
        # use getattr method, which results in a recursive loop
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def batch_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def apply(self, fn: Callable, **kwargs) -> ArrayDict:
        return ArrayDict(
            (
                (
                    field,
                    (
                        arr.apply(fn, **kwargs)
                        if hasattr(arr, "apply")
                        else fn(arr, **kwargs)
                    ),
                )
                for field, arr in self.items()
            ),
            _run_checks=False,
        )

    def to_ndarray(self) -> ArrayDict[np.ndarray]:
        return self.apply(to_ndarray)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value.to_dict() if isinstance(value, ArrayDict) else value
            for key, value in self._dict.items()
        }

    map = apply


def to_ndarray(leaf: ArrayLike) -> np.ndarray | ArrayDict[np.ndarray]:
    if hasattr(leaf, "to_ndarray"):
        return leaf.to_ndarray()
    if hasattr(leaf, "numpy"):
        # torch Tensor
        return leaf.numpy()
    return np.asarray(leaf)


class ArrayAttrDict(ArrayDict):
    def __init__(self, *args, name: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        items = []
        for field, method in self._dict.items():
            try:
                result = method(*args, **kwds)
            except Exception as e:
                if not callable(method):
                    raise RuntimeError(
                        f"Attribute '{self.name}' of field '{field}' is not callable!"
                    ) from e

                raise RuntimeError(
                    f"Exception from calling method '{self.name}' of field '{field}'"
                ) from e

            items.append((field, result))

        return ArrayDict(items)
