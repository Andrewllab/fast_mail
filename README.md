# Fast-MaIL

This is an open-source project that aims to provide a set of imitation learning algorithms and environments.

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
  - [ ] refine framework, add trainers
  - [ ] add adaln layers to mamba (later the film condition should be on all methods)
### Xuan Zhao
- [x] add selection of multi processing (temporarily solve local object unpicklable error)
- [x] add testing progress & states visualization
- [x] add basic bc_xlstm sturcture, yaml file setting
- [ ] debug xlstm structure
- [ ] multi processing model save/load debug

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
conda create -n mail python=3.11
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
```
### install xlstm
```
git clone https://github.com/NX-AI/xlstm.git
cd xlstm
pip install -e .
```
---

## Acknowledgements

The code of this repository relies on the following existing codebases:
- [Mamba] https://github.com/state-spaces/mamba
- [xLSTM] https://github.com/NX-AI/xlstm
- [D3Il] https://github.com/ALRhub/d3il
- [LIBERO] https://github.com/Lifelong-Robot-Learning/LIBERO
