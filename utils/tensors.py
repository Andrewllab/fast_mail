from torch import Tensor


def unsqueeze_to(x: Tensor, target: Tensor) -> Tensor:
    """Appends dimensions to the end of a tensor until it has the same
    dimensionality as the target.
    """
    n_unsqueeze = max(0, target.ndim - x.ndim)
    return x[(...,) + (None,) * n_unsqueeze]
    return x[(...,) + (None,) * n_unsqueeze]
