import h5py


def recursive_hdf5_to_dict(hdf5_obj):
    """
    Recursively convert a h5py object to a dictionary.

    Args:
        hdf5_obj (h5py.Group or h5py.Dataset): The h5py object to convert.

    Returns:
        dict: The converted dictionary.
    """
    if isinstance(hdf5_obj, h5py.Group):
        return {key: recursive_hdf5_to_dict(hdf5_obj[key]) for key in hdf5_obj.keys()}
    elif isinstance(hdf5_obj, h5py.Dataset):
        return hdf5_obj[()]
    else:
        raise TypeError(f"Unsupported type: {type(hdf5_obj)}")