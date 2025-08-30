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

Install [pytorch](https://pytorch.org/get-started/locally/)>=2.7, torchvision and [torch geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html):

```bash
pip3 install torch torchvision torch_geometric
```

Install the additional libraries for torch geometric for your specific torch and cuda version according to [the instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#additional-libraries):

```bash
python -c "import torch; print(torch.__version__)"
# >>> 2.8.0+cu128

TORCH_PLUS_CUDA=$(python -c "import torch; print(torch.__version__)")
pip install torch_cluster torch_scatter -f https://data.pyg.org/whl/torch-${TORCH_PLUS_CUDA}.html
```

### Building torch_cluster and torch_scatter from source

If pip cannot find a precompiled binary of torch_cluster and torch_scatter for your specific pytorch+cuda versions, you must compile them from source.
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

pip install --no-build-isolation --verbose torch_cluster torch_scatter
```

The `--no-build-isolation` flag was necessary for me, but may not be necessary for you depending on your platform.

### Kluster

The kluster does not have cuda toolkit (or nvcc) installed, yet the binaries from pip depend on a glibc 2.32, while the system only has glibc 2.31.
Therefore, we need to install nvcc inside the conda environment instead.

```bash
mamba install -c nvidia cuda-nvcc=12.8 cuda-toolkit=12.8
pip install --no-binary=:all: --verbose torch-scatter torch-cluster
```

These commands worked for me, even though I did not set any of the required paths from the documentation.

## Python dependencies

Install the remaining requirements:

```bash
pip install -r requirements.txt
```

Don't forget to log in to wandb, especially if you are setting up on a cluster or new platform!
```bash
wandb login
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

Users with Ubuntu 22.04/24.04 can simply install IsaacSim and IsaacLab with pip:

```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
pip install isaaclab[isaacsim,all]==2.0.2 --extra-index-url https://pypi.nvidia.com
```

After installation, you can reinstall torch again, since isaaclab installs torch 2.5.1. Pip will complain about version mismatches, but you can ignore this.

```bash
pip3 install --upgrade torch torchvision torch_geometric
```

Be aware that isaaclab downgrades your numpy version to 1.26.4, and is **not** compatible with numpy 2.x.
For this and other reasons, it is not recommended to install IsaacLab in the same environment as Novometis, FoundationStereo, or any other optional dependency.

## FoundationStereo

Since FoundationStereo is very large, we compile it with TensorRT to speed up inference times. The following compilation steps have been tested on the following platforms:
- Ubuntu 22.04, RTX 3090, Nvidia driver 570 (CUDA 12.8)
- Ubuntu 22.04, RTX 5090, Nvidia driver 570 (CUDA 12.8)
- Pop! OS 22.04, RTX 5080, Nvidia driver 570 (CUDA 12.8)
- Ubuntu 24.04, RTX 5090, Nvidia driver 575 (CUDA 12.9)

1. Install the [nvidia CUDA network repo](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu). This is equivalent to the first 3 steps of [installing the CUDA toolkit](https://developer.nvidia.com/cuda-downloads) when choosing "deb (network)" as the installer type. For Ubuntu 22.04, it looks like this:
    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    ```
1. Users of CUDA 12.x need to pin the CUDA 12.x version of TensorRT and all related libraries, forcing apt to ignore newer versions.
    Query apt for all available versions of TensorRT, and find one that is compatible with CUDA 12.x.
    ```bash
    apt-cache madison tensorrt
    #  ...
    #  tensorrt | 10.13.2.6-1+cuda12.9 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  Packages
    #  ...
    ```
    Here, cuda12.9 means it supports CUDA 12.0-12.9, **not** that only CUDA 12.9 is supported.
    Copy the chosen version string below and add the pin:
    ```bash
    TRT_VERSION=10.13.2.6-1+cuda12.9  # copy the chosen version string here
    echo "Package: libnvinfer* libnvonnxparsers* tensorrt* python3-libnvinfer*" | sudo tee /etc/apt/preferences.d/tensorrt-cuda12-9.pref
    echo "Pin: version $TRT_VERSION" | sudo tee -a /etc/apt/preferences.d/tensorrt-cuda12-9.pref
    echo "Pin-Priority: 1001" | sudo tee -a /etc/apt/preferences.d/tensorrt-cuda12-9.pref
    ```

    If you don't do this step, apt will try to install the latest version, which uses CUDA 13.x and is not compatible with your graphics driver. If you only specify the exact version of TensorRT that you want, apt will try to install all dependencies for CUDA 13.x, and cannot resolve the request.
1. Install TensorRT.
    ```bash
    sudo apt install tensorrt
    ```
    Don't forget to add tensorrt to your path by adding the following to your `.bashrc` file.
    ```bash
    export PATH=${PATH:+${PATH}:}/usr/src/tensorrt/bin
    ```
1. Install additional pip dependencies in your mamba environment.
    ```bash
    pip install onnx tensorrt-cu12  # this installs the TensorRT bindings for CUDA 12.x
    pip install -r requirements_fs.txt
    ```
1. Download the pretrained models.
    ```bash
    gdown --folder -O ./foundation_stereo_models https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf
    ```
1. Make the onnx file. Adjust the image dimensions, batch size, and output filepath as you wish. This will take ~30 seconds.
    ```bash
    XFORMERS_DISABLED=1 python scripts/make_onnx.py \
            --save_path ./foundation_stereo_models/foundation_stereo_small_3x480x640.onnx \
            --ckpt_dir ./foundation_stereo_models/11-33-40/model_best_bp2.pth \
            --height 480 \
            --width 640 \
            --valid_iters 16 \
            --batch_size 3
    ```
1. Compile the onnx file into a TensorRT engine. This will take ~30 minutes.
    ```bash
    trtexec --onnx=foundation_stereo_models/foundation_stereo_small_3x480x640.onnx \
            --saveEngine=foundation_stereo_models/foundation_stereo_small_3x480x640.plan \
            --fp16 \
            --verbose
    ```

    `dynamic_axes` is currently disabled in the `make_onnx.py` script, but if these are enabled, then trtexec must be passed the `--optShapes` argument (e.g. `--optShapes=left:3x3x480x640,right:3x3x480x640` for a batch size of 3). Otherwise it always assumes a batch size of 1.

Note: according to [the documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#python-package-index-installation), installing both the apt and pip packages for TensorRT is redundant and "may not be desirable". However, I was not able to find another way. The `trtexec` executable is part of the `libnvinfer-bin` apt package, while the pip package is required due to `import tensorrt` statements in python. With more trial and error, it's probably possible to find a way to install all the required software in user space only.

### Troubleshooting FoundationStereo

- Try not to do anything that requires too much GPU memory during compilation. If compilation randomly crashes after 10+ minutes, this may be the cause.
- If the file `/usr/local/cuda/targets/x86_64-linux/lib/libnvinfer_builder_resource.so.10.9.0` exists on your system, this may be causing a problem. `trtexec` seems to want to load this dynamic library if it exists, even though it's (presumably) for an older version of `libnvinfer` (10.9 vs. 10.13). Uninstall all TensorRT and CUDA apt packages (carefully). You may see a message like `/usr/local/cuda not empty so not removing` after uninstalling CUDA. After all traces of CUDA should be gone from your system, if this file (and a few others) are still there, remove them with `rm -rf`. Reinstall TensorRT (CUDA toolkit is not required) and try compilation again.

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
