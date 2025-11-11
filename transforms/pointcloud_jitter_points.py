from __future__ import annotations


# BackCompat
def JitterPointCloud(specs, max_sigma, pcd_keys="pcd"):
    from transforms.jitter import RandomTranslationalJitter

    assert pcd_keys == "pcd"

    return RandomTranslationalJitter(specs, sigma=max_sigma, distribution="uniform")
