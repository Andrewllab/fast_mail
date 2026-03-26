from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Mapping, Protocol, TypeVar, Union

LeafType = TypeVar("LeafType")
Tree = Union[LeafType, Mapping[str, "Tree[LeafType]"]]


def flatten_tree(
    tree: Mapping[str, "Tree[LeafType]"], sep: str = ".", parent_key: str = ""
) -> Mapping[str, "Tree[LeafType]"]:
    """
    Recursively flatten a nested dictionary.

    Example:
        {"a": {"b": 1, "c": 2}, "d": 3}
        -> {"a.b": 1, "a.c": 2, "d": 3}

    Args:
        tree: The nested dictionary to flatten.
        sep: Separator to place between nested keys.
        parent_key: Prefix for recursive calls.

    Returns:
        A flat dictionary.
    """
    items = {}

    for key, value in tree.items():
        new_key = parent_key + sep + key if parent_key else key

        if isinstance(value, Mapping):
            items.update(flatten_tree(value, parent_key=new_key, sep=sep))
        else:
            items[new_key] = value

    try:
        # attempt to construct a mapping of the same type as the input,
        # e.g. TensorDict, OrderedDict, etc.
        return type(tree)(items)  # type: ignore
    except (TypeError, ValueError):
        # otherwise just return a regular dict
        return items


def unflatten_tree(
    tree: Mapping[str, "Tree[LeafType]"], separator: str = "."
) -> Mapping[str, "Tree[LeafType]"]:
    """
    Converts a flat dictionary with delimited keys into a nested dictionary.
    """
    items = {}

    for key, value in tree.items():
        parts = key.split(separator)
        current_level = items

        # Iterate through the keys except for the very last one
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

        # Assign the value to the final key
        current_level[parts[-1]] = value

    try:
        # attempt to construct a mapping of the same type as the input,
        # e.g. TensorDict, OrderedDict, etc.
        return type(tree)(items)  # type: ignore
    except (TypeError, ValueError):
        # otherwise just return a regular dict
        return items


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
ReturnArrayType = TypeVar("ReturnArrayType", bound=ArrayLike)
ArrayTree = Union[ArrayType, None, Mapping[str, "ArrayTree[ArrayType]"]]


def tree_map(
    func: Callable[[ArrayType], ReturnArrayType],
    tree: ArrayTree[ArrayType],
    *args,
    **kwargs,
) -> ArrayTree[ReturnArrayType]:
    if isinstance(tree, Mapping):  # non-leaf node
        items = {
            key: tree_map(func, value, *args, **kwargs) for key, value in tree.items()
        }
        try:
            # attempt to construct a mapping of the same type as the input,
            # e.g. TensorDict, OrderedDict, etc.
            return type(tree)(items)  # type: ignore
        except (TypeError, ValueError):
            # otherwise just return a regular dict
            return items

    if tree is None:
        return None

    # leaf node
    return func(tree, *args, **kwargs)


def tree_call_method(
    tree: ArrayTree[ArrayType],
    method_name: str,
    *args,
    **kwargs,
) -> ArrayTree[ArrayType]:
    if isinstance(tree, Mapping):  # non-leaf node
        items = {
            key: tree_call_method(value, method_name, *args, **kwargs)
            for key, value in tree.items()
        }

        try:
            # attempt to construct a mapping of the same type as the input,
            # e.g. TensorDict, OrderedDict, etc.
            return type(tree)(items)  # type: ignore
        except (TypeError, ValueError):
            # otherwise just return a regular dict
            return items

    if tree is None:
        return None

    # leaf node
    method = getattr(tree, method_name)
    return method(*args, **kwargs)


def tree_get_item(
    tree: ArrayTree[ArrayType],
    idx: Any,
) -> ArrayTree[ArrayType]:
    if isinstance(tree, Mapping):  # non-leaf node
        items = {key: tree_get_item(value, idx) for key, value in tree.items()}

        try:
            # attempt to construct a mapping of the same type as the input,
            # e.g. TensorDict, OrderedDict, etc.
            return type(tree)(items)  # type: ignore
        except (TypeError, ValueError):
            # otherwise just return a regular dict
            return items

    if tree is None:
        return None

    # leaf node
    return tree[idx]


def tree_set_item(
    tree: ArrayTree[ArrayType],
    idx: Any,
    value: ArrayTree[ArrayType],
) -> None:
    if isinstance(tree, Mapping):  # non-leaf node
        for key, arr in tree.items():
            tree_set_item(arr, idx, value[key])
        return

    if tree is None:
        return

    # leaf node
    tree[idx] = value


def tree_all(
    tree: ArrayTree[ArrayType], predicate: Callable[[ArrayType], bool]
) -> bool:
    if isinstance(tree, Mapping):  # non-leaf node
        return all(tree_all(elem, predicate) for elem in tree.values())

    if tree is None:
        return True

    # leaf node
    return predicate(tree)
