from utils.figures.robocasa import _init_env
from PIL import Image
from agents.utils.sim_path import sim_framework_path
import visualizer

env_name = "PnPCounterToMicrowave"

env = _init_env(
        env_name=env_name,
        img_width=512,
        img_height=512,
        render=True,
        pc_num_points=1024,
        obj_max_num_points=512,
        use_segmented_point_cloud=False,
        seed=42,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    )

obs = env.reset()

lang = env.get_ep_meta()["lang"]

print(lang)

sampled_point_cloud = obs["sampled_point_cloud"]
full_point_cloud = obs["point_cloud"]

visualizer.visualize_pointcloud(sampled_point_cloud)

# left_img = Image.fromarray(obs["robot0_agentview_left_image"])
# right_img = Image.fromarray(obs["robot0_agentview_right_image"])
# eye_in_hand_img = Image.fromarray(obs["robot0_eye_in_hand_image"])
#
# left_img.save(sim_framework_path(f"utils/figures/{env_name}_left_img.png"))
# right_img.save(sim_framework_path(f"utils/figures/{env_name}_right_img.png"))
# eye_in_hand_img.save(sim_framework_path(f"utils/figures/{env_name}_eye_in_hand_img.png"))

# for i in range(10000):
#     env.render()