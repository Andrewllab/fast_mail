"""
Synchronous script to extract/re-render observations from low-dimensional simulation states
in pre-recorded RoboCasa dataset.

Adapted from robocasa/scripts/dataset_states_to_obs.py
"""

import argparse
import json
import os
import time
import traceback
from collections import OrderedDict
from copy import deepcopy

import h5py
import numpy as np
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_tensor_utils as TensorUtils
from robosuite.models.tasks.task import get_subtree_geom_ids_by_group
from robosuite.utils import camera_utils
from tqdm import tqdm


def record_cam_params(env, cam_names, W, H):
    """Returns {cam: {intrinsics (3x3), extrinsics (4x4 world frame), width, height}}"""
    sim = env.base_env.sim
    camera_params = {}
    for cam in cam_names:
        camera_intrinsics = camera_utils.get_camera_intrinsic_matrix(sim, cam, H, W)
        camera_extrinsics = camera_utils.get_camera_extrinsic_matrix(sim, cam)
        camera_params[cam] = {
            "intrinsics": camera_intrinsics.astype(np.float32),
            "extrinsics": camera_extrinsics.astype(np.float32),
            "width": int(W),
            "height": int(H),
        }
    return camera_params


def extract_trajectory(
    env, initial_state, states, actions, done_mode, args, add_datagen_info=False
):
    """
    Extract observations, rewards, dones along a trajectory by loading simulator states.

    done_mode:
      0: done=1 when success state
      1: done=1 at end of episode
      2: both
    """
    assert states.shape[0] == actions.shape[0]

    env.reset()
    env.reset_to(initial_state)

    ep_meta = env.env.get_ep_meta()
    initial_state["ep_meta"] = json.dumps(ep_meta, indent=4)

    W = int(args.camera_width)
    H = int(args.camera_height)
    camera_names = list(args.camera_names)

    default_dynamic_camera_names = [
        "robot0_agentview_left",
        "robot0_agentview_right",
        # "robot0_eye_in_hand",
    ]
    dynamic_cam_names = [c for c in default_dynamic_camera_names if c in camera_names]
    static_cam_names = [c for c in camera_names if c not in dynamic_cam_names]

    dynamic_cam_logs = {
        name: {"intrinsics": [], "extrinsics": []} for name in dynamic_cam_names
    }
    static_cam_logs = {
        name: {"intrinsics": [], "extrinsics": []} for name in static_cam_names
    }

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        states=np.array(states),
        initial_state_dict=initial_state,
        datagen_info=[],
        dynamic_cameras=[],
        static_cameras=[],
    )

    traj_len = states.shape[0]

    for t in tqdm(range(traj_len), desc="Extracting"):
        obs = deepcopy(env.reset_to({"states": states[t]}))

        # flip depth + segmentation; rgb already handled upstream in robocasa wrapper
        for key in obs:
            if "depth" in key:
                obs[key] = np.flip(obs[key], axis=0)
                obs[key] = camera_utils.get_real_depth_map(env.base_env.sim, obs[key])
            if "segmentation_element" in key:
                obs[key] = np.flip(obs[key], axis=0)

        if t == 0 and static_cam_names:
            static_camera_params = record_cam_params(env, static_cam_names, W, H)
            for name in static_cam_names:
                static_cam_logs[name]["extrinsics"].append(
                    static_camera_params[name]["extrinsics"]
                )

        if dynamic_cam_names:
            cam_parameters = record_cam_params(env, dynamic_cam_names, W, H)
            for name in dynamic_cam_names:
                dynamic_cam_logs[name]["extrinsics"].append(
                    cam_parameters[name]["extrinsics"]
                )

        datagen_info = (
            env.base_env.get_datagen_info(action=actions[t]) if add_datagen_info else {}
        )

        r = env.get_reward()

        done = False
        if (done_mode == 1) or (done_mode == 2):
            done = done or (t == traj_len - 1)  # FIX: was (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            done = done or env.is_success()["task"]
        done = int(done)

        segmentation_ids = {}
        for body_key in env.env.obj_body_id.keys():
            obj_body_id = env.env.obj_body_id[body_key]
            segmentation_ids[body_key] = [
                geom_id
                for geom_id in range(env.env.sim.model.ngeom)
                if env.env.sim.model.geom_bodyid[geom_id] == obj_body_id and geom_id
            ]
            segmentation_ids[body_key] += get_subtree_geom_ids_by_group(
                env.env.sim.model, obj_body_id, group=1
            )

        for fixture_key in env.env.fixtures_id.keys():
            fixture_body_id = env.env.fixtures_id[fixture_key]
            segmentation_ids[fixture_key] = [
                geom_id
                for geom_id in range(env.env.sim.model.ngeom)
                if env.env.sim.model.geom_bodyid[geom_id] == fixture_body_id and geom_id
            ]
            segmentation_ids[fixture_key] += get_subtree_geom_ids_by_group(
                env.env.sim.model, fixture_body_id, group=1
            )

        traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["datagen_info"].append(datagen_info)
        traj["segmentation_ids"] = segmentation_ids

    # intrinsics once (don’t change)
    if dynamic_cam_names:
        cam_parameters = record_cam_params(env, dynamic_cam_names, W, H)
        for name in dynamic_cam_names:
            dynamic_cam_logs[name]["intrinsics"].append(
                cam_parameters[name]["intrinsics"]
            )

    if static_cam_names:
        cam_parameters = record_cam_params(env, static_cam_names, W, H)
        for name in static_cam_names:
            static_cam_logs[name]["intrinsics"].append(
                cam_parameters[name]["intrinsics"]
            )

    if static_cam_names:
        static_cam_params = {}
        for name in static_cam_names:
            intrinsics = np.stack(static_cam_logs[name]["intrinsics"], axis=0).astype(
                np.float32
            )
            extrinsics = np.stack(static_cam_logs[name]["extrinsics"], axis=0).astype(
                np.float32
            )
            static_cam_params[name] = dict(
                intrinsics=intrinsics, extrinsics=extrinsics, width=W, height=H
            )
        traj["static_cameras"].append(static_cam_params)

    if dynamic_cam_names:
        dynamic_cam_params = {}
        for name in dynamic_cam_names:
            intrinsics = np.stack(dynamic_cam_logs[name]["intrinsics"], axis=0).astype(
                np.float32
            )
            extrinsics = np.stack(dynamic_cam_logs[name]["extrinsics"], axis=0).astype(
                np.float32
            )
            dynamic_cam_params[name] = dict(
                intrinsics=intrinsics, extrinsics=extrinsics, width=W, height=H
            )
        traj["dynamic_cameras"].append(dynamic_cam_params)

    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["datagen_info"] = TensorUtils.list_of_flat_dict_to_dict_of_list(
        traj["datagen_info"]
    )

    # convert lists to numpy arrays
    for k in list(traj.keys()):
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in list(traj[k].keys()):
                if isinstance(traj[k][kp], dict):
                    traj[k][kp] = TensorUtils.list_of_flat_dict_to_dict_of_list(
                        traj[k][kp]
                    )
                    for kpp in list(traj[k][kp].keys()):
                        traj[k][kp][kpp] = np.array(traj[k][kp][kpp])
                else:
                    traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def write_one_episode(ep, traj, f_src, data_grp_out, args):
    """Write a single episode group into the output file."""
    ep_data_grp = data_grp_out.create_group(ep)

    dataset_cfgs = {"compression": "gzip"}
    if args.no_compress:
        dataset_cfgs = {}

    # fmt: off
    ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]), **dataset_cfgs)
    ep_data_grp.create_dataset("states", data=np.array(traj["states"]), **dataset_cfgs)
    ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]), **dataset_cfgs)
    ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]), **dataset_cfgs)
    # fmt: on

    # Write obs using require_group so intermediate groups exist
    obs_root = ep_data_grp.require_group("obs")

    for k in traj["obs"]:
        if isinstance(traj["obs"][k], OrderedDict) or isinstance(traj["obs"][k], dict):
            k_grp = obs_root.require_group(str(k))
            for kp in traj["obs"][k]:
                data = np.array(traj["obs"][k][kp])
                k_grp.create_dataset(str(kp), data=data, **dataset_cfgs)
        else:
            data = np.array(traj["obs"][k])
            obs_root.create_dataset(str(k), data=data, **dataset_cfgs)

    if args.include_next_obs and "next_obs" in traj and len(traj["next_obs"]) > 0:
        next_root = ep_data_grp.require_group("next_obs")
        for k in traj["next_obs"]:
            data = np.array(traj["next_obs"][k])
            next_root.create_dataset(str(k), data=data, **dataset_cfgs)

    # camera params
    if "static_cameras" in traj and len(traj["static_cameras"]) > 0:
        for camera_name, cam_dict in traj["static_cameras"][0].items():
            cam_grp = ep_data_grp.require_group(f"camera_params/static/{camera_name}")
            for camera_data_key, camera_data_value in cam_dict.items():
                cam_grp.create_dataset(
                    camera_data_key, data=camera_data_value, **dataset_cfgs
                )

    if "dynamic_cameras" in traj and len(traj["dynamic_cameras"]) > 0:
        for camera_name, cam_dict in traj["dynamic_cameras"][0].items():
            cam_grp = ep_data_grp.require_group(f"camera_params/dynamic/{camera_name}")
            for camera_data_key, camera_data_value in cam_dict.items():
                cam_grp.create_dataset(camera_data_key, data=camera_data_value)

    # datagen info
    if "datagen_info" in traj and isinstance(traj["datagen_info"], dict):
        di_grp = ep_data_grp.require_group("datagen_info")
        for k in traj["datagen_info"]:
            di_grp.create_dataset(
                str(k), data=np.array(traj["datagen_info"][k]), **dataset_cfgs
            )

    # segmentation ids
    if "segmentation_ids" in traj and isinstance(traj["segmentation_ids"], dict):
        seg_grp = ep_data_grp.require_group("segmentation_ids")
        for class_name in traj["segmentation_ids"]:
            seg_grp.create_dataset(
                str(class_name),
                data=np.array(traj["segmentation_ids"][class_name]),
                **dataset_cfgs,
            )

    # copy action dict (if applicable)
    if f"data/{ep}/action_dict" in f_src:
        action_dict = f_src[f"data/{ep}/action_dict"]
        ad_grp = ep_data_grp.require_group("action_dict")
        for k in action_dict:
            ad_grp.create_dataset(
                str(k), data=np.array(action_dict[k][()]), **dataset_cfgs
            )

    # episode metadata
    ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
    ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"]["ep_meta"]
    ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]

    return traj["actions"].shape[0]


def dataset_states_to_obs_sync(args):
    # output file in same directory as input file
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            image_suffix = str(args.camera_width)
            image_suffix = (
                image_suffix + "_randcams" if args.randomize_cameras else image_suffix
            )
            if args.generative_textures:
                output_name = (
                    os.path.basename(args.dataset)[:-5]
                    + f"_gentex_im{image_suffix}.hdf5"
                )
            else:
                output_name = (
                    os.path.basename(args.dataset)[:-5] + f"_im{image_suffix}.hdf5"
                )

    output_path = os.path.join(os.path.dirname(args.dataset), output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"input file:  {args.dataset}")
    print(f"output file: {output_path}")

    # Read demo list
    with h5py.File(args.dataset, "r") as f:
        if args.filter_key is not None:
            print(f"using filter key: {args.filter_key}")
            demos = [
                elem.decode("utf-8") for elem in np.array(f[f"mask/{args.filter_key}"])
            ]
        else:
            demos = list(f["data"].keys())

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    if args.n is not None:
        demos = demos[: args.n]

    # Build env once
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True

    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )

    total_samples = 0
    start_time = time.time()

    with h5py.File(args.dataset, "r") as f_src, h5py.File(output_path, "w") as f_out:
        data_grp_out = f_out.create_group("data")

        for i, ep in enumerate(demos, start=1):
            try:
                ep_grp = f_src[f"data/{ep}"]

                states = ep_grp["states"][()]
                actions = ep_grp["actions"][()]

                initial_state = dict(states=states[0])
                initial_state["model"] = ep_grp.attrs["model_file"]
                initial_state["ep_meta"] = ep_grp.attrs.get("ep_meta", None)

                traj = extract_trajectory(
                    env=env,
                    initial_state=initial_state,
                    states=states,
                    actions=actions,
                    done_mode=args.done_mode,
                    add_datagen_info=args.add_datagen_info,
                    args=args,
                )

                if args.copy_rewards and "rewards" in ep_grp:
                    traj["rewards"] = ep_grp["rewards"][()]
                if args.copy_dones and "dones" in ep_grp:
                    traj["dones"] = ep_grp["dones"][()]

                n_written = write_one_episode(ep, traj, f_src, data_grp_out, args)
                total_samples += n_written

                rate = (time.time() - start_time) / i
                print(
                    f"ep {i}/{len(demos)}: wrote {n_written} transitions to group {ep}. "
                    f"Rate: {rate:.2f} sec/demo"
                )

            except Exception as e:
                print("_" * 80)
                print(f"Error processing episode {ep}: {e}")
                print(traceback.format_exc())
                print("_" * 80)
                raise

        data_grp_out.attrs["total"] = total_samples

    # Post-processing on the finished dataset
    DatasetUtils.extract_action_dict(dataset=output_path)
    for num_demos in [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        75,
        80,
        90,
        100,
        125,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1500,
        2000,
        2500,
        3000,
        4000,
        5000,
        10000,
    ]:
        DatasetUtils.filter_dataset_size(output_path, num_demos=num_demos)

    elapsed = time.time() - start_time
    print(f"Wrote {total_samples} total samples to {output_path}")
    print(f"Time elapsed: {elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="path to input hdf5 dataset"
    )
    parser.add_argument("--output_name", type=str, help="name of output hdf5 dataset")
    parser.add_argument("--filter_key", type=str, help="filter key for input dataset")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )
    parser.add_argument(
        "--shaped", action="store_true", help="(optional) use shaped rewards"
    )

    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            # "robot0_eye_in_hand",
        ],
        help="camera name(s) to use for image observations",
    )
    parser.add_argument("--camera_height", type=int, default=224)
    parser.add_argument("--camera_width", type=int, default=224)

    parser.add_argument("--done_mode", type=int, default=0)
    parser.add_argument("--copy_rewards", action="store_true")
    parser.add_argument("--copy_dones", action="store_true")
    parser.add_argument("--include-next-obs", action="store_true")
    parser.add_argument("--no_compress", action="store_true")
    parser.add_argument("--add_datagen_info", action="store_true")
    parser.add_argument("--generative_textures", action="store_true")
    parser.add_argument("--randomize_cameras", action="store_true")

    args = parser.parse_args()
    dataset_states_to_obs_sync(args)
