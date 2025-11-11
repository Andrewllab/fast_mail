from __future__ import annotations


# BackCompat
def TranslationalJitter(specs, sigma):
    from transforms.jitter import RandomTranslationalJitter

    return RandomTranslationalJitter(
        specs, sigma=sigma, distribution="normal", random_sigma=False
    )


# BackCompat
def VariableTranslationalJitter(specs, max_sigma):
    from transforms.jitter import RandomTranslationalJitter

    return RandomTranslationalJitter(
        specs, sigma=max_sigma, distribution="normal", random_sigma=True
    )
