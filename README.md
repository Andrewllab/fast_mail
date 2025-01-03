# Fast-MaIL

This is an open-source project that aims to provide a set of imitation learning algorithms and environments.

---

## Desired Features

- **Observations:** support for both state and visual observations, as well as language embeddings.
- **Encoders:** support for mlp, resnet, vit and other pretrained models.
- **Architectures:** support for transformer, mamba, xlstm models.
- **Policy Head:** support for bc, ddpm, beso, flow matching and vqbet.
- **Environments:** support for push-T, block-push, libero, and other tasks.

- **Potential:** think about adding point cloud inputs, maybe try 1 or 2 tasks and real robot

---

## Installation

To begin, clone this repository locally
```
git clone git@github.com:xiaogangjia/fast_mail.git
```

### Installing requirements
```
conda create -n mail python=3.10
conda activate mail

# adapt to your own cuda version if you need
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

### Installing LIBERO Setup
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

### Installing RoboCasa Setup
```
git clone https://github.com/robocasa/robocasa
cd robocasa
pip install -e .
python robocasa/scripts/download_kitchen_assets.py
```

### Installing packages for beso
```
pip install torchsde torchdiffeq
```

### Installing packages for mamba1/2
```
pip install mamba-ssm

# if the above command doesn't work, check if the torch cuda version mathes and try the following
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e .

# if you still have errors
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install setuptools==61.0.0
python setup.py install
pip install causal-conv1d>=1.4.0

# in horeka, the following command works
pip install mamba-ssm[causal-conv1d] --no-build-isolation
```
### install xlstm
```
git clone https://github.com/NX-AI/xlstm.git
cd xlstm
pip install -e .
```
### installing packages for equibot
```
conda install -y fvcore iopath ffmpeg -c iopath -c fvcore
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade setuptools wheel
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install diffusers
```

### Installing point cloud visualizer
```
cd visualizer
pip install -e .
```
---

## Acknowledgements

The code of this repository relies on the following existing codebases:
- [Mamba] https://github.com/state-spaces/mamba
- [xLSTM] https://github.com/NX-AI/xlstm
- [D3Il] https://github.com/ALRhub/d3il
- [LIBERO] https://github.com/Lifelong-Robot-Learning/LIBERO


---
## TODO

### Xiaogang Jia
- 12.10 -> basic structure implementation
  - [x] Implement the basic structure of the project.
  - [x] Libero dataset and environment.
  - [x] BC policy.
  - [x] Transformer (encoder_decoder)
  - [x] Basic training scripts and evaluation
- 12.11 -> beso and mamba1/2
  - [x] add beso (with mdt architecture)
  - [x] add mamba1/2
- 12.12
  - [x] refine framework, add trainers
- 12.17
  - [x] unify decoder-only and encoder-decoder
  - [ ] add adaln layers to mamba (later the film condition should be on all methods)
  - [ ] add beso mamba
  - [ ] unify transformer
  - [ ] provide language encoder and if use language text
  - [ ] write different encoder-decoder structures
### Xuan Zhao
- [x] Add progress tracking and state visualization
- [x] Implement BC-XLSTM structure with YAML config
- [x] Debug XLSTM implementation
- [x] Fix multiprocessing model save/load; add multiprocessing selection (unpicklable error)
- [x] VQBET
- [ ] Clean up code

### Misc
- [ ] Standardize camera image observations across LIBERO and RoboCasa
- [ ] Resolve XLSTM context length issues
- [ ] Add custom XLSTM block stacking functionality