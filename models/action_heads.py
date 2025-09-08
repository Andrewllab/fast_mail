import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


class FourierActions(nn.Module):

    wavelengths: Tensor

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        mlp: Callable[[int, int], nn.Linear],
        n_wavelengths: int,
        max_wavelength: float,
        min_wavelength: float,
        n_encoded_actions: int | None = None,
    ) -> None:
        # TODO: implement different numerical approximations to asin(tanh(x))
        super().__init__()

        self.action_dim = action_dim
        self.n_wavelengths = n_wavelengths
        self.n_encoded_actions = n_encoded_actions if n_encoded_actions else action_dim

        # wavelengths decreasing exponentially from max_wavelength to min_wavelength
        exponents = torch.linspace(
            start=math.log(max_wavelength),
            end=math.log(min_wavelength),
            steps=n_wavelengths,
        )
        wavelengths = torch.exp(exponents)

        wavelengths = wavelengths / torch.pi

        self.register_buffer("wavelengths", wavelengths)

        self.mlp = mlp(
            in_features,
            self.n_encoded_actions * self.n_wavelengths
            + (action_dim - self.n_encoded_actions),
        )

    def forward(self, features: Tensor) -> Tensor:
        # fourier_actions: (B, T, D)
        features = self.mlp(features)

        fourier_features = features[..., : self.n_encoded_actions * self.n_wavelengths]

        # (B, T, D) -> (B, T, A, N)
        fourier_features = fourier_features.unflatten(
            dim=-1, sizes=(self.n_encoded_actions, self.n_wavelengths)
        )

        # bound fourier features to [-1, 1]
        fourier_features = torch.tanh(fourier_features)

        # pos = arcsin(θ) / π
        pos = torch.asin(fourier_features)

        pos = pos * self.wavelengths

        # (B, T, A, N) -> (B, T, A)
        pos = pos.sum(dim=-1)

        return torch.cat(
            (pos, features[..., self.n_encoded_actions * self.n_wavelengths :]), dim=-1
        )

    @property
    def in_features(self) -> int:
        """Returns the input size of the model."""
        return self.mlp.in_features

    @property
    def out_features(self) -> int:
        """Retuns the output size of the model."""
        return self.action_dim


import torch
from torch.autograd import Function


class ArcSinhTanh(Function):
    @staticmethod
    def forward(ctx, input):
        # Use stable gudermannian: 2 * atan(tanh(x/2))
        half = input.mul(0.5)
        t = torch.tanh(half)
        out = 2.0 * torch.atan(t)
        # Save tanh(x) for backward (we need tanh(x) to compute sech(x))
        # tanh(x) can be recovered from tanh(x/2) via identity:
        # tanh(x) = 2*tanh(x/2) / (1 + tanh(x/2)^2)
        tanh_half = t
        tanh_full = 2.0 * tanh_half / (1.0 + tanh_half * tanh_half)
        ctx.save_for_backward(tanh_full)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (tanh_full,) = ctx.saved_tensors
        # sech(x) = sqrt(1 - tanh(x)^2)
        # Ensure numerical safety: clamp inside sqrt to [0,1]
        inside = 1.0 - tanh_full * tanh_full
        inside_clamped = torch.clamp(inside, min=0.0, max=1.0)
        sech = torch.sqrt(inside_clamped)
        grad_input = grad_output * sech
        return grad_input


# Convenience wrapper
def arcsin_tanh(x):
    return ArcSinhTanh.apply(x)


# Quick test
if __name__ == "__main__":
    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        x = torch.tensor(
            [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0], device=device, requires_grad=True
        )
        y = arcsin_tanh(x)
        print(f"device={device} y = {y}")

        # Check analytical backward (sech(x)) against autograd numeric gradcheck
        y_sum = y.sum()
        y_sum.backward()
        print("grad (from custom backward) =", x.grad)

        # finite-difference check
        eps = 1e-4
        fd = []
        x_np = x.detach().cpu().numpy()
        for i in range(x.numel()):
            x_plus = x.detach().clone()
            x_minus = x.detach().clone()
            x_plus.view(-1)[i] += eps
            x_minus.view(-1)[i] -= eps
            y_plus = arcsin_tanh(x_plus).detach().cpu().numpy()
            y_minus = arcsin_tanh(x_minus).detach().cpu().numpy()
            fd.append(((y_plus - y_minus) / (2 * eps)).reshape(-1)[i])
        print("finite-diff grads =", torch.tensor(fd, device=device))
