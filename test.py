from collections import deque
from copy import deepcopy
import h5py
from robocasa.utils.env_utils import create_env
from environments.dataset.robocasa_dataset import RobocasaDataset
from environments.wrappers.robosuite_wrapper import RobosuiteWrapper
from robosuite.utils.transform_utils import *
from tqdm import tqdm

dataset_path = "/home/i53/student/donat/master_thesis/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/processed_demo_2.hdf5"

f = h5py.File(dataset_path, "r")

env_name = "PnPCounterToCab"
env = create_env(
    env_name=env_name,
    camera_widths=640,
    camera_heights=360,
    camera_names=["robot0_agentview_center"],
    render_onscreen=True,
)

env = RobosuiteWrapper(env)

ds = RobocasaDataset(cam_names=["robot0_agentview_center"], data_directory=dataset_path, device="cpu", action_dim=7, obs_dim=13, max_len_data=260, window_size=16)   
# i = 0

for demo_name in f["data"]:
# for demo_name in ["demo_10"]:
    demo = f["data"][demo_name]
    initial_state = dict(states=demo["states"][0])
    initial_state["model"] = demo.attrs["model_file"]
    initial_state["ep_meta"] = demo.attrs.get("ep_meta", None)
    obs = deepcopy(env.reset_to(initial_state))

    env.render()

    for i in tqdm(range(len(demo["actions"]))):
        # if rollout_step_counter == 0:
        # obs_dict, action, mask = ds[i]
            # pred_action_seq = action[:act_seq_len]
            
        # action = action[0].numpy()
        action = demo["global_actions"][i]
        
        # action = pred_action_seq[rollout_step_counter].numpy()
        # action = np.concatenate(
        #     [action[1:], action[:1], np.array([0, 0, 0, 0, -1])]
        # )

        base_mat = env.sim.data.get_site_xmat(f"mobilebase{env.robots[0].idn}_center")
        
        global_action_pos = action[:3]
        global_action_axis_angle = action[3:6]
        global_action_mat = quat2mat(axisangle2quat(global_action_axis_angle))
        
        action[:3] = base_mat.T @ global_action_pos
        local_action_mat = base_mat.T @ global_action_mat @ base_mat
        action[3:6] = quat2axisangle(mat2quat(local_action_mat))
                
        obs, reward, done, info = env.step(action)
        env.render()
        
        # i += 1