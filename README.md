# Fast-MaIL

This is an open-source project that aims to provide a set of imitation learning algorithms and environments.


## Installation

To begin, clone this repository locally
```
git clone --recursive git@github.com:balazsgyenes/fast_mail.git
```

### Dependencies

Create a new virtual environment and install requirements:
```
conda create -p .env python=3.11
conda activate ./.env

pip install -r requirements.txt
```

Note: if you use a different CUDA version, please adapt the requirements file accordingly.


### Data

To download the original furniture bench dataset, please follow the [furniture bench documentation](https://clvrai.github.io/furniture-bench/docs/tutorials/dataset.html).

Our custom datasets are located on the lsdf. Consider mounting the lsdf using sshfs and just linking the data where you want it to be available.

By default, the training data should be located in a subfolder called `datasets`. However, this can be modified by creating a local override config. For example, create a file at `configs/local/default.yaml`, and add the following:

```yaml
# @package _global_

# path to data directory
paths:
  data_dir: ${oc.env:HOME}/datasets

```


## Acknowledgements

The code of this repository is based on the [Fast-MaIL framework](https://github.com/xiaogangjia/fast_mail) of Xiaogang Jia.
