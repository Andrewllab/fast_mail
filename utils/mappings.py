from typing import Literal, Mapping, MutableMapping, TypeVar

T = TypeVar("T", bound=Mapping)


def unnest_dict(d: T, parent_key: str = "", sep: str = "/") -> T:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if bool(parent_key) and bool(sep) else k
        if isinstance(v, Mapping):
            items.extend(unnest_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    # this weird syntax also works for TensorDict
    return type(d)(dict(items))


U = TypeVar("U", bound=MutableMapping)


def accumulate_dict(d1: U, d2: Mapping, aggr: Literal["sum", "max"] = "sum") -> U:
    for key, value in d2.items():
        if key not in d1:
            d1[key] = value
        else:
            if aggr == "sum":
                d1[key] += value
            elif aggr == "max":
                d1[key] = max(d1[key], value)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")

    return d1
