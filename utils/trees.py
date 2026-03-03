from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Mapping, Protocol, TypeVar, Union


class ArrayLike(Protocol):
    def __getitem__(self, indices: Any, /) -> Any:
        # / is to indicate that the parameter names do not matter
        # https://stackoverflow.com/questions/75420105/python-typing-callback-protocol-and-keyword-arguments
        ...

    def __setitem__(self, indices: Any, value: Any, /) -> None: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...


ArrayType = TypeVar("ArrayType", bound=ArrayLike)
ArrayTree = Union[ArrayType, None, Mapping[str, Union[ArrayType, None, "ArrayTree"]]]


def tree_map(
    func: Callable[[ArrayType], Any],
    tree: ArrayTree[ArrayType],
    *args,
    **kwargs,
) -> ArrayTree:
    if isinstance(tree, Mapping):  # non-leaf node
        return type(tree)(
            (key, tree_map(func, value, *args, **kwargs)) for key, value in tree.items()
        )

    if tree is None:
        return None  # type: ignore

    # leaf node
    return func(tree, *args, **kwargs)


def tree_call_method(
    tree: ArrayTree[ArrayType],
    method_name: str,
    *args,
    **kwargs,
) -> ArrayTree:
    if isinstance(tree, Mapping):  # non-leaf node
        return type(tree)(
            (key, tree_call_method(value, method_name, *args, **kwargs))
            for key, value in tree.items()
        )

    if tree is None:
        return None  # type: ignore

    # leaf node
    method = getattr(tree, method_name)
    return method(*args, **kwargs)


def tree_get_item(
    tree: ArrayTree[ArrayType],
    idx: Any,
) -> ArrayTree:
    if isinstance(tree, Mapping):  # non-leaf node
        return type(tree)(
            ((key, tree_get_item(value, idx)) for key, value in tree.items())
        )

    if tree is None:
        return None

    # leaf node
    return tree[idx]


def tree_set_item(
    tree: ArrayTree[ArrayType],
    idx: Any,
    value: ArrayType,
) -> None:
    if isinstance(tree, Mapping):  # non-leaf node
        for key, arr in tree.items():
            tree_set_item(arr, idx, value[key])
        return

    if tree is None:
        return

    # leaf node
    tree[idx] = value


def tree_all(tree: ArrayTree, predicate: Callable[[ArrayLike], bool]) -> bool:
    if isinstance(tree, Mapping):  # non-leaf node
        return all(tree_all(elem, predicate) for elem in tree.values())

    if tree is None:
        return True

    # leaf node
    return predicate(tree)
