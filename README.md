# Fast-MaIL

This is an open-source project that aims to provide a set of imitation learning algorithms and environments.


## Installation

To begin, clone this repository locally
```bash
git clone --recursive git@github.com:balazsgyenes/fast_mail.git
```

Create a conda/mamba environment with **Python 3.10** (required for IsaacLab):
```bash
conda create -p ./.env python=3.10
conda activate ./.env
```

### Torch

Most users can install the stable version of [pytorch](https://pytorch.org/get-started/locally/) and [torch geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html):

```bash
pip3 install torch torchvision torch_geometric
```

Install the additional libraries for torch geometric for your specific torch and cuda version according to [the instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#additional-libraries):

```bash
python -c "import torch; print(torch.__version__)"
# >>> 2.6.0
# TORCH=2.6.0

python -c "import torch; print(torch.version.cuda)"
# >>> 12.6
# CUDA=cu126

pip install torch_cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

**RTX 50 Series Users**

Users with RTX 50 series GPUs must install the nightly release of pytorch (2.8.*).
This requires installing the additional libraries from source:

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install torch_geometric
```

Install the CUDA toolkit version that matches the CUDA version of your installed pytorch. You can check the version like this:

```bash
python -c "import torch; print(torch.version.cuda)"
```

After installing CUDA toolkit, don't forget to add the cuda/bin folder to your path according to the [post-installation actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#mandatory-actions)!
Then run the following commands according to [the instructions from torch geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-source):

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

pip install --no-build-isolation --verbose torch_cluster
```

### Python dependencies

Install the remaining requirements:
```bash
pip install -r requirements.txt
```

### Optional dependencies

Other (optional) dependencies can be installed as follows:
```bash
pip install -r requirements_robot.txt  # for execution on the real robot
pip install -r requirements_test.txt  # for running tests, etc.
```

## Data

To download the original furniture bench dataset, please follow the [furniture bench documentation](https://clvrai.github.io/furniture-bench/docs/tutorials/dataset.html).

Our custom datasets are located on the lsdf. Consider mounting the lsdf using sshfs and just linking the data where you want it to be available.

By default, the training data should be located in a subfolder called `datasets`. However, this can be modified by creating a local override config. For example, create a file at `configs/local/default.yaml`, and add the following:

```yaml
# @package _global_

# path to data directory
paths:
  data_dir: ${oc.env:HOME}/datasets

```

## Usage

### Useful Commands

Start training on dataset DATA (optional), for experiment EXP (optional), on platform PLAT (optional):

```bash
python train.py data=DATA experiment=EXP platform=PLAT
```

Open loop replay using dataset DATA:

```bash
python predict.py -cn=open_loop_replay data@agent.replay_data=DATA
```

Visualize dataset DATA (optional), after applying transforms for experiment EXP. You must also choose whether to render RGB images with render_cameras or pointcloud with render_pointcloud. Simultaneously rendering both is not possible.

```bash
python predict.py -cn visualize_dataset "experiment=[EXP,render_cameras]" data=DATA
python predict.py -cn visualize_dataset "experiment=[EXP,render_pointcloud]" data=DATA
```


## Acknowledgements

The code of this repository is based on the [Fast-MaIL framework](https://github.com/xiaogangjia/fast_mail) of Xiaogang Jia.
