# Fast-MaIL

This is an open-source project that aims to provide a set of imitation learning algorithms and environments.


# Installation

To begin, clone this repository locally
```bash
git clone --recursive git@github.com:balazsgyenes/fast_mail.git
```

Create a conda/mamba environment with **Python 3.10** (required for IsaacLab):
```bash
mamba create -p ./.env python=3.10
mamba activate ./.env
```

## Torch

Most users can install the stable version of [pytorch](https://pytorch.org/get-started/locally/) and [torch geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html):

```bash
pip3 install torch torchvision torch_geometric
```

Install the additional libraries for torch geometric for your specific torch and cuda version according to [the instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#additional-libraries):

```bash
python -c "import torch; print(torch.__version__)"
# >>> 2.8.0+cu128

TORCH_PLUS_CUDA=$(python -c "import torch; print(torch.__version__)")
pip install torch_cluster -f https://data.pyg.org/whl/torch-${TORCH_PLUS_CUDA}.html
pip install torch_scatter -f https://data.pyg.org/whl/torch-${TORCH_PLUS_CUDA}.html
```

### Horeka

Horeka only supports CUDA versions 12.4 and 12.9, whereas the default is 12.8, so run the following:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

module load devel/cuda/12.9

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

Depending on your use case, there may also be additional pip requirements to install:

### Horeka

Horeka requires the submitit launcher for hydra.

```bash
pip install -r requirements_horeka.txt
```

### Real Robot

The real robot requires some additional hardware drivers and opencv.

```bash
pip install -r requirements_robot.txt
```

Also don't forget to install the pyzed using the following command:

```bash
python /usr/local/zed/get_python_api.py
```

### Testing

Some testing code and mockups require pytest or other packages.

```bash
pip install -r requirements_test.txt
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

# Data

To download the original furniture bench dataset, please follow the [furniture bench documentation](https://clvrai.github.io/furniture-bench/docs/tutorials/dataset.html).

Our custom datasets are located on the lsdf. For best performance, download the dataset to prevent repeatedly accessing the data over the network. If you're short on disk space, you can mount the lsdf using sshfs and just link the data where you want it to be available.

By default, the training data should be located in a subfolder called `datasets`. However, this can be modified by creating a local override config. For example, create a file at `configs/local/default.yaml`, and add the following:

```yaml
# @package _global_

# path to data directory
paths:
  data_dir: ${oc.env:HOME}/datasets/3d-sim2real
  preprocessed_dir: ${oc.env:HOME}/preprocessed_data
  debug_preprocessed_dir: ${oc.env:HOME}/.cache/debug_preprocessed_data
```

## Horeka

Horeka can be very very slow when it comes to file I/O. The fastest way to get data into a training run on the cluster is to do the preprocessing locally, compress it, transfer the archive, and then only uncompress inside the job itself.

1. On horeka, create a file at `configs/local/default.yaml`, and add the following (or something similar):
    ```yaml
    # @package _global_

    # path to data directory
    paths:
      zipped_preprocessed_dir: ${oc.env:HOME}/zipped_preprocessed_datasets
    ```
1. On your local machine, run training with debug config `preprocess` and ensure that your data is preprocessed as you want it.
    ```bash
    # e.g. if you want to train with pointclouds on the cluster
    python train.py obs_modality=sim_pointclouds obs_encoder=dp3 debug=preprocess
    ```
1. On your local machine, compress the preprocessed dataset using tar and zstd (this uses all your cores, unlike gzip). Ensure that the internal paths are relative to the root folder of the preprocessed dataset and the archive name is the folder name of the dataset. Below is an example for the `metaquest3_2025-07-24` dataset with `multiview_image` preprocessing.
    ```bash
    mkdir -pv ~/zipped_preprocessed_datasets/metaquest3_2025-07-24
    # -C is the folder that internal paths should be relative to
    # -f is the output archive
    tar -I "zstd -T0" -cv \
      -f ~/zipped_preprocessed_datasets/metaquest3_2025-07-24/multiview_image.tar.zst \
      -C ~/preprocessed_data/metaquest3_2025-07-24/multiview_image .
    ```
1. From your local machine, use rsync (or similar) to transfer the data to horeka into the ${paths.zipped_preprocessed_data} folder you defined in the config file above:
    ```bash
    # first make sure this folder exists on the remote system
    ssh horeka "mkdir -pv ~/zipped_preprocessed_datasets/metaquest3_2025-07-24"
    rsync -hP --no-owner --no-group \
      ~/zipped_preprocessed_datasets/metaquest3_2025-07-24/multiview_image.tar.zst \
      horeka:zipped_preprocessed_datasets/metaquest3_2025-07-24/
    ```
1. To start training on horeka, just modify the platform config like so:
    ```bash
    # add any other arguments as desired,
    python train.py platform=horeka
    ```

**Tip**: If you use `less` like I do, use `less -R` to look at slurm logs on horeka, otherwise ANSI escape sequences (for colored output) will not be displayed properly.


# Useful Commands

## Training

Start training with default settings for everything:

```bash
python train.py ~debug
```

Start training on dataset DATA (optional), for experiment EXP (optional), on platform PLAT (optional):

```bash
python train.py data=DATA experiment=EXP platform=PLAT ~debug
```

Start training without logging to wandb and without saving model checkpoints:

```bash
python train.py debug="[no_wandb,no_model_checkpoints]"
```

## Testing

Test a trained model (from local log_dir PATH) by computing MSE against held-out demonstration data:

```bash
python test.py --config-name=test_on_dataset log_dir=PATH
```

Test a trained model (from wandb run with id ID) on real robot:

```bash
python predict.py --config-name=test_real artifact_run_name=ID
```

Test a trained model (from wandb run with id ID) on an IsaacLab environment: 
```bash 
python predict.py --config-name=test_isaac_lab artifact_run_name=ID
```

Test a trained model (from wandb run with id ID) on an IsaacLab environment and record a video of each episode: 
```bash 
python predict.py --config-name=test_isaac_lab artifact_run_name=ID data.env_dataset.env.wrapper_cfgs.names="[RecordVideo,IsaacLabPreProcess]"
```

## Misc

Visualize dataset DATA (optional), after applying transforms for experiment EXP. You must also choose whether to render RGB images with render_cameras or pointcloud with render_pointcloud. Simultaneously rendering both is not possible.

```bash
# to render images:
python predict.py --config-name=visualize_dataset +transforms@agent.obs_encoder.t9_render=render_cameras
# to render point clouds:
python predict.py --config-name=visualize_dataset obs_modality=sim_pointclouds +transforms@agent.obs_encoder.t9_render_pcd=render_pointcloud
```

Open loop replay on real robot using dataset DATA:

```bash
python predict.py --config-name=open_loop_replay data@agent.replay_data=DATA
```

Visualize observations produced by vision pipeline for experiment EXP on real robot:

```bash
python predict.py --config-name=visualize_real_robot experiment=EXP
```


In case there is a problem about using a determinisic environment set: CUBLAS_WORKSPACE_CONFIG=:4096:8

Control the robot with teleoperation with e.g. keyboard in IsaacLab environment: 
```bash 
python scripts/teleop_se3_agent.py --teleop_device keyboard --task Isaac-Insert-One-Leg-Franka-IK-Rel-v0
```

# Acknowledgements

The code of this repository is based on the [Fast-MaIL framework](https://github.com/xiaogangjia/fast_mail) of Xiaogang Jia.
