import time
import torch
from mamba_ssm import Mamba2
from mamba_ssm import Mamba

repeat_num = 1000
batch, length, dim = 2, 256*256*2, 256
x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=64,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# y = model(x) # warm up first
# t1 = time.time()
# for i in range(repeat_num):
#     y = model(x)
# assert y.shape == x.shape
# print(f"Time of mamba taken: {time.time() - t1:.3f} s")

model = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=64,  # SSM state expansion factor, typically 64 or 128
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x) # warm up first
t1 = time.time()
for i in range(repeat_num):
    y = model(x)
assert y.shape == x.shape
print(f"Time of mamba2 taken: {time.time() - t1:.3f} s")