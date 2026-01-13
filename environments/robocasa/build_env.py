import functools
import logging

import hydra
from omegaconf import DictConfig

from environments.robocasa.wrappers import GymWrapper, RoboCasaPreProcess

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
    """
    import robocasa
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        # robosuite-related configs
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        camera_depths=True,  # render RGBD
        has_renderer=render_onscreen,  # whether to render onscreen
        has_offscreen_renderer=(not render_onscreen),  # whether to render headless
        ignore_done=True,  # no timeout
        use_object_obs=True,  # add proprioception to observation
        use_camera_obs=(not render_onscreen),  # whether to add rendering to obs
        seed=seed,
        # robocasa-related configs
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


def make(
    env_name: str,
    img_height: int,
    img_width: int,
    seed: int | None = None,
    wrappers: DictConfig | None = None,
):
    """
    Creates a RoboCasa environment and applies specified wrappers using Hydra.

    Args:
        env_name: The environment name of the RoboCasa environment to create.
        img_size: Width and height of the observations.
        camera_names: List of camera names used for observations.
        wrappers: A DictConfig from Hydra containing wrapper configurations.
                  Each wrapper must have '_target_' and '_partial_: True'.
    """

    log.info(f"Building RoboCasa environment: '{env_name}'")

    env = create_env(
        env_name=env_name,
        camera_widths=img_width,
        camera_heights=img_height,
        seed=seed,
    )

    log.info("Instantiating wrappers...")
    wrappers_partials = hydra.utils.instantiate(wrappers)

    if wrappers_partials is None:
        return env

    # filter out any values that are not partials
    wrappers_partials = [
        wrapper
        for wrapper in wrappers_partials.values()
        if isinstance(wrapper, functools.partial)
    ]

    # find the GymWrapper wrapper and move it to the front of the list
    gym_wrappers = [i for i, w in enumerate(wrappers_partials) if w.func is GymWrapper]
    if len(gym_wrappers) > 1:
        raise ValueError("Only one GymWrapper is allowed.")
    elif len(gym_wrappers) == 1:
        i = gym_wrappers[0]
        gym_wrapper = wrappers_partials.pop(i)
        wrappers_partials.insert(0, gym_wrapper)

    # find the RoboCasaPreProcess wrapper and move it to the end of the list
    preprocess_wrappers = [
        i for i, w in enumerate(wrappers_partials) if w.func is RoboCasaPreProcess
    ]
    if len(preprocess_wrappers) > 1:
        raise ValueError("Only one RoboCasaPreProcess is allowed.")
    elif len(preprocess_wrappers) == 1:
        i = preprocess_wrappers[0]
        preprocess_wrapper = wrappers_partials.pop(i)
        wrappers_partials.append(preprocess_wrapper)

    # apply wrappers in succession
    for wrapper in wrappers_partials:
        env = wrapper(env)

    log.info("Finished applying all wrappers.")

    return env
