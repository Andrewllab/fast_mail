import torch


def compute_model_fft(
    model: torch.nn.Module,
    point: torch.Tensor,  # shape (3)
    direction: torch.Tensor,  # shape (..., 3)
    min_wavelength: float,
    n_steps: int = 512,
    n_bins: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = min_wavelength / 2  # sampling at twice the Nyquist frequency

    # (n_steps,)
    displacements = (torch.arange(n_steps, device=point.device) - n_steps // 2) * delta

    # normalize direction (..., 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    direction = direction.unsqueeze(dim=-2)  # -> (..., 1, 3)

    # (..., n_steps, 3)
    displaced_points = point + displacements.unsqueeze(dim=-1) * direction

    with torch.no_grad():
        features = model(displaced_points)  # (..., n_steps, D)

    # (n_steps // 2 + 1, D)
    fft_features = torch.fft.rfft(
        features,
        dim=-2,  # along the n_steps dimension
        # without normalization, the amplitude scales linearly with n_steps
        norm="forward",
    )

    # compute the magntitude to get the amplitude of each frequency in the
    # signal
    fft_features = fft_features.abs()
    # remove constant component, (..., n_steps // 2, D)
    fft_features = fft_features[..., 1:, :]

    # # transpose so that the frequency dimension is last
    # fft_features = fft_features.transpose(-1, -2)  # (..., D, n_steps // 2 + 1)

    # (n_steps // 2 + 1,)
    freqs = torch.fft.rfftfreq(n_steps, d=delta, device=point.device)
    freqs = freqs[1:]  # remove constant component

    if n_bins is not None:
        # create logarithmically spaced frequency bins
        # start by computing the midpoints of the bins
        log_freqs = freqs.log()
        log_freq_ticks = torch.linspace(
            log_freqs.min(), log_freqs.max(), steps=n_bins, device=point.device
        )

        # compute the edges of each bin
        log_freq_incr = log_freq_ticks[1] - log_freq_ticks[0]
        log_freq_bins = torch.linspace(
            log_freqs.min() - log_freq_incr / 2,
            log_freqs.max() + log_freq_incr / 2,
            steps=n_bins + 1,
            device=point.device,
        )

        # assign each frequency to a bin
        idx = torch.bucketize(log_freqs, log_freq_bins, right=False) - 1
        idx = idx.clamp(min=0, max=n_bins - 1)

        # find empty bins and merge them with the bin on their left
        bin_counts = torch.bincount(idx, minlength=n_bins)
        empty_bins = (bin_counts == 0).nonzero().squeeze(-1)
        mask = torch.ones_like(log_freq_bins, dtype=torch.bool)
        mask[empty_bins] = False
        mask[0] = True  # always keep first bin
        # last bin is always kept because we have one more bin edge than bins
        log_freq_bins = log_freq_bins[mask]

        # recompute midpoints of bins
        n_bins = len(log_freq_bins) - 1
        log_freq_ticks = (log_freq_bins[:-1] + log_freq_bins[1:]) / 2.0
        # rebin frequencies
        idx = torch.bucketize(log_freqs, log_freq_bins, right=False) - 1
        idx = idx.clamp(min=0, max=n_bins - 1)

        # sum amplitudes within each frequency bin
        leading_dims = fft_features.shape[:-2]
        D = fft_features.shape[-1]
        binned_shape = leading_dims + (n_bins, D)
        # (..., n_directions, n_bins, D)
        binned_fft = fft_features.new_zeros(binned_shape)
        # broadcast idx to match the shape of fft_features for scatter_add
        idx = idx[(None,) * len(leading_dims) + (..., None)].expand(fft_features.shape)
        binned_fft.scatter_add_(dim=-2, index=idx, src=fft_features)

        return torch.exp(log_freq_ticks), torch.exp(log_freq_bins), binned_fft

    freq_incr = freqs[1] - freqs[0]
    freq_bins = torch.linspace(
        freqs.min() - freq_incr / 2,
        freqs.max() + freq_incr / 2,
        steps=freqs.shape[0] + 1,
        device=point.device,
    )

    return freqs, freq_bins, fft_features
