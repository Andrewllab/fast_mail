import functools
import logging
from typing import Any, List, Mapping

from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import TimeLimit

from environments.wrappers import (
    ActionChunkWrapper,
    DtypeObservation,
    NumpyToTorch,
    RecordEpisodeStatistics,
    RecordMultiEpisodeVideo,
    RemoveInfoMasks,
)

log = logging.getLogger(__name__)


def create_env(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    camera_depths=False,
    seed=None,
    render_onscreen=False,
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
):
    """Copied and adapted from robocasa.utils.env_utils.create_env

    Changes from default version:
    - always render depth when rendering cameras
    - disable use_camera_obs so we can control when to render cameras ourselves
    """
    import robocasa
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(
        controller=None,
        robot="PandaOmron",
    )

    env_kwargs = dict(
        env_name=env_name,
        robots="PandaOmron",
        controller_configs=controller_config,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        camera_depths=True,  # render RGBD
        has_renderer=False,  # do not render in a viewer
        has_offscreen_renderer=True,  # do render headless
        ignore_done=True,  # no timeout
        use_object_obs=True,  # add proprioception to observation
        use_camera_obs=True,  # add rendered camera images to each observation
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    return env


def make_one(
    env_name: str,
    img_height: int,
    img_width: int,
    camera_names: List[str],
    seed: int | None = None,
    max_episode_steps: int | float | None | Mapping[str, int] = None,
    render_cam_name: str | None = "robot0_agentview_left",
    render_size: tuple[int, int] | None = (256, 256),
):
    """
    Creates a RoboCasa environment and applies specified wrappers using Hydra.

    Args:
        env_name: The environment name of the RoboCasa environment to create.
        img_height: Height of the observations.
        img_width: Width of the observations.
        seed: Random seed for environment initialization.
        max_episode_steps: Maximum number of steps per episode. If a mapping
            from env_name to int is provided, the value corresponding to the
            env_name is used, or the value corresponding to "default" if
            env_name is not in the mapping. If None, the environment's
            registered "horizon" value is used as the time limit.
    Returns:
        A Gymnasium environment with the specified wrappers applied.
    """
    import robocasa  # register environments
    from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS
    from robosuite.wrappers import GymWrapper

    from environments.robocasa.wrappers import RoboCasaAdapter

    log.info(f"Building RoboCasa environment: '{env_name}'")

    import robocasa
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(
        controller=None,
        robot="PandaOmron",
    )

    env_kwargs = dict(
        env_name=env_name,
        # robosuite-related configs
        robots="PandaOmron",
        controller_configs=controller_config,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=img_width,
        camera_heights=img_height,
        has_renderer=False,  # do not render in a viewer
        has_offscreen_renderer=True,  # do render headless
        ignore_done=True,  # no timeout
        use_object_obs=True,
        use_camera_obs=False,  # do not render all steps, just the ones we want
        camera_depths=True,  # render RGBD
        seed=seed,
        # robocasa-related configs
        obj_instance_split=None,
        generative_textures=None,
        randomize_cameras=False,
        layout_and_style_ids=None,
        layout_ids=None,
        style_ids=None,
        translucent_robot=False,
    )

    camera_names = env.camera_names
    proprio_keys = ["joint_pos", "gripper_qpos", "eef_pos", "eef_quat"]

    # TODO: don't hardcode the streams here
    obs_keys = (
        [f"robot0_{key}" for key in proprio_keys]
        + [f"{cam}_image" for cam in camera_names]
        + [f"{cam}_depth" for cam in camera_names]
    )

    # Create Gymnasium observation space and action space, and filter a subset
    # of the observation keys.
    env = GymWrapper(env, keys=obs_keys, flatten_obs=False)

    # Add a time limit.
    if isinstance(max_episode_steps, Mapping):
        if env_name in max_episode_steps:
            max_episode_steps = max_episode_steps[env_name]
        else:
            max_episode_steps = max_episode_steps.get("default", None)

    if max_episode_steps is None:
        try:
            max_episode_steps = SINGLE_STAGE_TASK_DATASETS[env_name]["horizon"]
            log.info(
                f"Using default timeout of {max_episode_steps} steps for environment '{env_name}'"
            )
        except KeyError:
            raise KeyError(
                f"max_episode_steps is not provided and env_name '{env_name}' is not in SINGLE_STAGE_TASK_DATASETS, so we cannot determine the time limit for the environment. Please provide max_episode_steps as an int or a mapping from env_name to int."
            )

    if not (isinstance(max_episode_steps, int) or max_episode_steps == float("inf")):
        raise ValueError(
            f"max_episode_steps must be None, an int or a mapping from env_name to int, but got {max_episode_steps}"
        )

    if max_episode_steps > 0 and max_episode_steps != float("inf"):
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # Flip camera images vertically, convert depth to real depth, add camera
    # extrinsics to the observation, and remove unused dimensions from the
    # actions.
    render_size = render_size or (img_height, img_width)
    env = RoboCasaAdapter(
        env,
        reduced_action_space=True,
        render_cam_name=render_cam_name,
        render_size=render_size,
    )

    return env


def make(
    env_name: str,
    img_height: int,
    img_width: int,
    seed: int | None = None,
    num_envs: int = 1,
    parallel: bool | None = False,
    action_horizon: int | None = None,
    obs_seq_len: int = 1,
    max_episode_steps: int | None = None,
    record_video: Mapping[str, Any] | None = None,
    render_cam_name: str | None = "robot0_agentview_left",
    render_size: tuple[int, int] | None = (256, 256),
):

    from environments.robocasa.wrappers import RoboCasaSpecs

    env_fn = functools.partial(
        make_one,
        env_name=env_name,
        img_height=img_height,
        img_width=img_width,
        seed=seed,
        max_episode_steps=max_episode_steps,
        render_cam_name=render_cam_name,
        render_size=render_size,
    )

    if parallel is None:
        parallel = num_envs > 1

    if parallel:
        VecEnvCls = functools.partial(
            AsyncVectorEnv,
            shared_memory=True,
            # avoid rendering issues in subprocesses by using spawn instead of fork
            context="spawn",
        )
    else:
        VecEnvCls = SyncVectorEnv

    envs = VecEnvCls(
        [env_fn for _ in range(num_envs)],
        # do not copy observations since we don't modify them in-place anywhere
        copy=False,
        # all environments are the same, so they have the same observation space
        observation_mode="same",
        # we will handle auto-resetting ourselves in the ActionChunkWrapper
        autoreset_mode=AutoresetMode.DISABLED,
    )

    # Gymnasium's VectorEnv adds a mask for each field of the info dict to
    # indicate which envs' info dicts contains that field. Since all environments
    # are the same and always contain all fields, we can remove these.
    envs = RemoveInfoMasks(envs)

    # Convert numpy arrays to torch tensors, because all vectorized wrappers
    # expect torch tensors.
    envs = NumpyToTorch(envs)

    # Convert proprioceptive observations from float64 to float32.
    envs = DtypeObservation(
        envs,
        filter_dtype="float64",
        target_dtype="float32",
    )

    # Record episode statistics such as success and episode length in the info
    # dict under the "episode" key.
    envs = RecordEpisodeStatistics(envs, stats_key="episode")

    record_video = record_video or {}
    record_video = dict(record_video)  # copy to ensure mutability
    if record_video and not record_video.pop("disabled", False):
        envs = RecordMultiEpisodeVideo(envs, **record_video)

    # Handle action chunking and auto-resetting when any env is done.
    envs = ActionChunkWrapper(
        envs,
        action_horizon=action_horizon,
        obs_seq_len=obs_seq_len,
        auto_reset=True,  # reset envs when done
    )

    # Add DataSpecs derived from observation and action spaces.
    envs = RoboCasaSpecs(envs)

    return envs
