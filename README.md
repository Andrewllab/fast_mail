# Fast-MaIL

This is an open-source project that aims to provide a set of imitation learning algorithms and environments.


# Installation

To begin, clone this repository locally
```bash
git clone --recursive git@github.com:balazsgyenes/fast_mail.git
```

Create a conda/mamba environment with **Python 3.10** (required for IsaacLab):
```bash
conda create -p ./.env python=3.10
conda activate ./.env
```

## Torch

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

TORCH_PLUS_CUDA=$(python -c "import torch; print(torch.__version__)")
pip install torch_cluster -f https://data.pyg.org/whl/torch-${TORCH_PLUS_CUDA}.html
pip install torch_scatter -f https://data.pyg.org/whl/torch-${TORCH_PLUS_CUDA}.html
```

### RTX 50 Series Users

Users with RTX 50 series GPUs must install the nightly release of pytorch (2.8.*).
This requires installing the additional libraries from source:

```bash
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
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
pip install --no-build-isolation --verbose torch_scatter
```

In case pip refuses to compile these packages from source, run:

```bash
pip install --no-cache --no-build-isolation --verbose torch_cluster
pip install --no-cache --no-build-isolation --verbose torch_scatter
```

## Python dependencies

Install the remaining requirements:

```bash
pip install -r requirements.txt
```

## IsaacLab

Users with Ubuntu 22.04 can simply install IsaacSim and IsaacLab with pip:

```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
pip install isaaclab[isaacsim,all]==2.0.2 --extra-index-url https://pypi.nvidia.com
```

**Ubuntu 20.04 Users**

Download and install the pre-built binaries for IsaacSim according to [the instructions](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html).
Add the required environment variables to your `.bashrc` file.
With your conda environment deactivated, verify that IsaacSim runs as expected using the commands [here](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#verifying-the-isaac-sim-installation).

```bash
# note: you can pass the argument "--help" to see all arguments possible.
${ISAACSIM_PATH}/isaac-sim.sh
# checks that python path is set correctly
${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
# checks that Isaac Sim can be launched from python
${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.api/add_cubes.py
```

Clone the IsaacLab repository and symlink IsaacSim into the IsaacLab directory:

```bash
git clone git@github.com:isaac-sim/IsaacLab.git
# enter the cloned repository
cd IsaacLab
# create a symbolic link
ln -s ${ISAACSIM_PATH} _isaac_sim
```

Install the isaaclab and isaaclab_tasks packages directly using pip:

```bash
pip install source/isaaclab
pip install source/isaaclab_tasks
```

## Optional dependencies

Other (optional) dependencies can be installed as follows:
```bash
pip install -r requirements_robot.txt  # for execution on the real robot
pip install -r requirements_test.txt  # for running tests, etc.
```

# Data

To download the original furniture bench dataset, please follow the [furniture bench documentation](https://clvrai.github.io/furniture-bench/docs/tutorials/dataset.html).

Our custom datasets are located on the lsdf. Consider mounting the lsdf using sshfs and just linking the data where you want it to be available.

By default, the training data should be located in a subfolder called `datasets`. However, this can be modified by creating a local override config. For example, create a file at `configs/local/default.yaml`, and add the following:

```yaml
# @package _global_

# path to data directory
paths:
  data_dir: ${oc.env:HOME}/datasets

```

# Usage

## Useful Commands

Start training on dataset DATA (optional), for experiment EXP (optional), on platform PLAT (optional):

```bash
python train.py data=DATA experiment=EXP platform=PLAT
```

Visualize dataset DATA (optional), after applying transforms for experiment EXP. You must also choose whether to render RGB images with render_cameras or pointcloud with render_pointcloud. Simultaneously rendering both is not possible.

```bash
python predict.py -cn visualize_dataset "experiment=[EXP,render_cameras]" data=DATA
python predict.py -cn visualize_dataset "experiment=[EXP,render_pointcloud]" data=DATA
```

Open loop replay on real robot using dataset DATA:

```bash
python predict.py -cn=open_loop_replay data@agent.replay_data=DATA
```

Test a trained model from wandb run with id ID on real robot:

```bash
python predict.py -cn=test_real artifact_run_name=ID
```

Visualize observations produced by vision pipeline for experiment EXP on real robot:

```bash
python predict.py -cn=visualize_real_robot experiment=EXP
```

Test a trained model on an IsaacLab environment using the wandb-id: 
  ```bash 
  python predict.py --cn test_isaac_lab artifact_run_name: 47v5jb3c
  ```

Test a trained model on an IsaacLab environment using the wandb-id and record a video of each episode: 
  ```bash 
  python predict.py --cn test_isaac_lab data.env_dataset.env.wrapper_cfgs.names=[RecordVideo,IsaacLabPreProcess] artifact_run_name: 47v5jb3c
  ```
In case there is a problem about using a determinisic environment set: CUBLAS_WORKSPACE_CONFIG=:4096:8

Control the robot with teleoperation with e.g. keyboard in IsaacLab environment: 
```bash 
python scripts/teleop_se3_agent.py --teleop_device keyboard --task Isaac-Insert-One-Leg-Franka-IK-Rel-v0
```

# Acknowledgements

The code of this repository is based on the [Fast-MaIL framework](https://github.com/xiaogangjia/fast_mail) of Xiaogang Jia.
