import torch
import tensorrt as trt
from collections import namedtuple, OrderedDict

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform
from utils.FoundationStereo.utils import InputPadder

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

class FoundationStereo(Transform):
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, specs: DataSpecs) -> None:

        self._ir_specs = [key for key, spec in specs.obs.items() if spec.type == "ir"] # TODO: Handle IR stereo obs correctly: { "ir": { "left": img1, "right": img2 } } ?
        self._specs = specs

        self.device = torch.device("cuda")

        self.context = self.engine.create_execution_context()

        self.ENGINE_PATH = "foundation_stereo_files/pretrained_models/foundation_stereo.engine" # TODO: make this configurable
        INPUT_NAME_0 = "left" # TODO: check if still necessary!
        INPUT_NAME_1 = "right" # TODO: check if still necessary!
        DTYPE = trt.float32 # TODO: check if still necessary!

        # Load tensorRT engine
        self.engine = load_engine(self.ENGINE_PATH.ENGINE_PATH)
        self.bindings = allocate_bindings(self.engine, self.device)

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("obs", key) for key in self._ir_specs],
                out_keys=[("obs", "depth_map")]) # TODO: check if naming is correct!
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def forward(self, ir_obs):

        H,W = ir_obs["left"].shape[:2]
        for side in ir_obs:
            ir_obs[side] = torch.as_tensor(ir_obs[side]).cuda().float()[None].permute(0,3,1,2) # TODO: get rid of the cuda() calls? (possible through lightning?)
        padder = InputPadder(ir_obs["left"].shape, divis_by=32, force_square=False) # TODO: should i also initialize the padder in __init__? For this i need the ir_obs["left"].shape!
        ir_obs["left"], ir_obs["right"] = padder.pad(ir_obs["left"], ir_obs["right"])

        # foundation stereo
        with torch.cuda.amp.autocast(True):
            run_inference(self.engine, self.context, self.bindings)

        output_binding = [name for name in self.bindings if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT][0]
        output = self.bindings[output_binding].data.cpu().numpy()
        disp = output.reshape(self.bindings[output_binding].shape)

        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H,W) # TODO: get rid of the cpu() calls? (possible through lightning?)
        return disp


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_bindings(engine, device):
    bindings = OrderedDict()
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            is_input = False
        else:
            is_input = True
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = tuple(engine.get_tensor_shape(name))
        if -1 in shape:
            raise ValueError(f"Dynamic shape for tensor '{name}': {shape}")
        data = torch.empty(size=shape, dtype=torch.float32, device=device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    return bindings

def run_inference(engine, context, bindings_dict):
    binding_addrs = [bindings_dict[engine.get_tensor_name(i)].ptr for i in range(engine.num_io_tensors)]
    context.execute_v2(binding_addrs)