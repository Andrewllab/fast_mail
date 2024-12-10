# Fast-MaIL

This is an open-source project that aims to provide a set of imitation learning algorithms and environments.

---

## Desired Features

- **Observations:** support for both state and visual observations, as well as language embeddings.
- **Encoders:** support for mlp, resnet, vit and other pretrained models.
- **Architectures:** support for transformer, mamba, xlstm models.
- **Policy Head:** support for bc, ddpm, beso, flow matching and vqbet.
- **Environments:** support for push-T, block-push, libero, and other tasks.

---
## TODO

### Xiaogang Jia
- 12.10 -> basic structure implementation
  - [x] Implement the basic structure of the project.
  - [x] Libero dataset and environment.
  - [x] DDPM and BC policy.
  - [x] Transformer and Mamba/Mamba2 model.
  - [x] Basic training scripts and evaluation
- evaluation and comparison

### Xuan Zhao


---

## Acknowledgements

The code of this repository relies on the following existing codebases:
- [Mamba] https://github.com/state-spaces/mamba
- [xLSTM] https://github.com/NX-AI/xlstm
- [D3Il] https://github.com/ALRhub/d3il
- [LIBERO] https://github.com/Lifelong-Robot-Learning/LIBERO
