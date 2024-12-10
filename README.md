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
- Next -> beso and mamba1/2

### Xuan Zhao


---

## Desired Features

- **Observations:** support for both state and visual observations, as well as language embeddings.
- **Encoders:** support for mlp, resnet, vit and other pretrained models.
- **Architectures:** support for transformer, mamba, xlstm models.
- **Policy Head:** support for bc, ddpm, beso, flow matching and vqbet.
- **Environments:** support for push-T, block-push, libero, and other tasks.

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

---

## Acknowledgements

The code of this repository relies on the following existing codebases:
- [Mamba] https://github.com/state-spaces/mamba
- [xLSTM] https://github.com/NX-AI/xlstm
- [D3Il] https://github.com/ALRhub/d3il
- [LIBERO] https://github.com/Lifelong-Robot-Learning/LIBERO
