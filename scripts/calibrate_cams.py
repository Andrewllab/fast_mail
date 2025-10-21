import logging
import os
import time
from pathlib import Path
from typing import Literal, Sequence

import cv2
import cv2.aruco as aruco
import hydra
import numpy as np
import open3d as o3d
import open3d.visualization as o3dvis
import pygame
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from environments.specs import DepthStream
from utils.calibration import (
    AcquisitionType,
    MetadataType,
    calibrate_intrinsics,
    compare_metadata,
    convert_extrinsics_convention,
    intrinsics_matrix_from_metadata,
    load_acquisitions,
    load_metadata,
    optimize_euclidean_dynamic_cam,
    optimize_reprojection,
    save_acquisition,
    save_metadata,
)
from utils.math import (
    convert_quat,
    make_pose,
    quaternion_to_matrix,
    transform_pointmap,
    unproject_depth,
)
from utils.rendering import (
    depth_to_renderable,
    intensity_to_renderable,
    rgb_to_renderable,
    tile_images,
)

log = logging.getLogger(__name__)

DEPTH_RENDER_DEFAULTS = {
    "colormap_name": "magma",
    "depth_min": 0.0,
    "depth_max": 2.0,
    "invert_colormap": False,
}


@hydra.main(version_base=None, config_path="../configs", config_name="calibrate_cams")
def main(cfg: DictConfig) -> None:

    do_collect = cfg.do_collect
    do_redetect_markers = cfg.do_redetect_markers
    do_recompute_depth = cfg.do_recompute_depth
    do_show_loaded = cfg.do_show_loaded
    do_render_fused_pcd = cfg.do_render_fused_pcd
    min_acquisitions = cfg.get("min_acquisitions", 4)

    do_save_calibration = (output_folder := cfg.get("output_folder")) is not None
    do_save_fused_pcd = (
        fused_pcd_save_path := cfg.get("fused_pcd_save_path")
    ) is not None
    has_foundation_stereo = (ckpt_path := cfg.get("foundation_stereo_ckpt")) is not None

    use_foundation_stereo = do_recompute_depth or (has_foundation_stereo and do_collect)
    use_pygame = do_collect or do_show_loaded

    # setup data folder
    acquisitions_folder = Path(cfg.acquisitions_folder)
    acquisitions_folder.mkdir(parents=True, exist_ok=True)

    acquisitions: list[AcquisitionType] = load_acquisitions(acquisitions_folder)

    metadata_file = acquisitions_folder / "camera_metadata.json"
    # if saved metadata exists, load it and compare
    if metadata_file.exists():
        log.info(f"Loading camera metadata from {metadata_file}")
        saved_metadata = load_metadata(metadata_file)

    if do_collect:
        from polymetis import RobotInterface

        from environments.real_robot.hardware.base_camera import BaseCamera
        from environments.real_robot.hardware.franka_control import HumanControl

        # connect to cameras
        cameras = hydra.utils.instantiate(cfg.cameras)
        cameras: list[BaseCamera] = list(cameras.values())

        # connect to robot arm (we don't need the gripper)
        robot = RobotInterface(
            name=cfg.robot.name,
            ip_address=cfg.robot.ip_address,
            port=cfg.robot.arm_port,
            enforce_version=False,
        )
        log.info(f'Connected to robot "{cfg.robot.name}" at {cfg.robot.ip_address}')
        if (home_pose := cfg.robot.get("home_pose", None)) is not None:
            log.info(f"Setting home pose: {home_pose}")
            robot.set_home_pose(torch.tensor(home_pose))
            robot.go_home()
        robot.send_torch_policy(HumanControl(robot), blocking=False)

        # find the streams which should be calibrated
        stream_names = cfg.get("stream_name", "left")
        if isinstance(stream_names, str):
            stream_names = [stream_names for _ in cameras]
        elif not isinstance(stream_names, list):
            raise ValueError(
                f"stream_name must be str or list, got {type(stream_names)}"
            )

        # find the names of the depth stream of each camera (probably "depth")
        depth_names = []
        for camera in cameras:
            names = [
                name
                for name, stream in camera.spec.streams.items()
                if isinstance(stream, DepthStream)
            ]
            # we assume one camera can have at most one depth stream
            assert len(names) == 1
            depth_names.append(names[0])

        # collect camera metadata (e.g. intrinsics, etc.)
        metadata = {}
        for camera, stream_name, depth_name in zip(cameras, stream_names, depth_names):
            metadata[camera.name] = {
                "camera_cls": f"{camera.__class__.__module__}.{camera.__class__.__name__}",
                "static": True,  # TODO: put this in config instead of hard-coding
                "stream_name": stream_name,
                "depth_stream_name": depth_name,
                "height_width": list(camera.spec.streams[stream_name].height_width),
                "intrinsics": camera.get_intrinsics(stream_name),
            }

        # if saved metadata exists, load it and compare, but do not overwrite it
        if metadata_file.exists():
            matches = compare_metadata(saved_metadata, metadata)
            if matches:
                log.debug("Loaded camera metadata matches current metadata")

        else:
            # save metadata to file
            save_metadata(metadata, metadata_file)
            log.info(f"Saved camera metadata to {metadata_file}")

    # metadata and previous acquisitions are required if not collecting
    elif not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file {metadata_file} does not exist!")
    elif not acquisitions:
        raise FileNotFoundError(f"No acquisitions found in {acquisitions_folder}!")
    else:
        metadata = saved_metadata

    # setup charuco board detector
    charuco_cfg = cfg.charuco
    # e.g. aruco.DICT_4X4_50
    aruco_dict = getattr(aruco, f"DICT_{charuco_cfg.aruco_dict}")
    aruco_dict = aruco.getPredefinedDictionary(aruco_dict)
    charuco_board = aruco.CharucoBoard(
        tuple(charuco_cfg.board_size),  # X, Y axis
        charuco_cfg.square_length,  # square length in meters
        charuco_cfg.marker_length,  # marker length (should be smaller!)
        aruco_dict,
    )
    charuco_detector = aruco.CharucoDetector(charuco_board)

    if do_redetect_markers:
        # re-detect markers in all acquisitions
        acquisitions = [
            detect_markers(acquisition, charuco_detector, charuco_board, metadata)
            for acquisition in acquisitions
        ]

    # setup foundation stereo
    if use_foundation_stereo:
        log.info(f"Loading FoundationStereo model from {ckpt_path}")
        model, padders = load_foundation_stereo(ckpt_path, metadata)

    if do_recompute_depth:
        # recompute depth with foundation stereo
        acquisitions = [
            run_foundation_stereo(acquisition, model, padders, metadata)
            for acquisition in acquisitions
        ]

    # define your calibration pipeline here
    def calibrate(
        acquisitions: Sequence[AcquisitionType], metadata: MetadataType
    ) -> tuple[np.ndarray, np.ndarray]:

        # don't calibrate intrinsics, just start with the factory calibration
        intrinsics = np.stack(
            [
                intrinsics_matrix_from_metadata(cam_metadata)
                for cam_metadata in metadata.values()
            ]
        )

        # optimize extrinsics so that the same point observed from all cameras
        # lines up in 3D space
        T_base2cam, T_base2obj = optimize_euclidean_dynamic_cam(
            acquisitions, intrinsics, metadata
        )

        # if desired, further optimize extrinsics (any maybe instrinsics) to
        # minimize reprojection error
        if cfg.optimization_objective == "euclidean":
            pass
        elif cfg.optimization_objective == "reprojection":
            intrinsics, T_base2cam, T_base2obj = optimize_reprojection(
                acquisitions,
                intrinsics,
                metadata,
                T_base2cam,
                T_base2obj,
                optimize_intrinsics=cfg.optimize_intrinsics,
            )
        else:
            raise ValueError(
                f"Unknown optimization objective '{cfg.optimization_objective}'"
            )

        return intrinsics, T_base2cam

    if len(acquisitions) >= min_acquisitions:
        # update calibration
        intrinsics, extrinsics = calibrate(acquisitions, metadata)
    else:
        intrinsics = np.stack(
            [
                intrinsics_matrix_from_metadata(cam_metadata)
                for cam_metadata in metadata.values()
            ]
        )
        # identity extrinsics
        extrinsics = np.stack([np.eye(4) for _ in metadata])

    # pygame setup
    if use_pygame:
        fps = cfg.fps
        pause_s = cfg.get("pause_s", 3.0)
        height, width = cfg.rendering.height_width
        n_height, n_width = 2, len(metadata)
        pygame.init()
        screen = pygame.display.set_mode((width * n_width, height * n_height))
        clock = pygame.time.Clock()

        render_cfg = {
            "height": height,
            "width": width,
            "n_height": n_height,
            "n_width": n_width,
        }
        depth_render_cfg = DEPTH_RENDER_DEFAULTS | {**cfg.rendering.get("depth", {})}

    if do_show_loaded:
        # visualize aquisitions
        for acquisition in acquisitions:
            rendering = render_acquisition(
                acquisition, metadata, render_cfg, depth_render_cfg
            )
            pygame.surfarray.blit_array(screen, rendering.transpose(1, 0, 2))
            pygame.display.flip()
            time.sleep(pause_s)  # pause so the user can see the final acquisition

        # reset the clock, so that the next interval starts from here
        clock.tick(fps)

    acquisition = None
    running = do_collect  # skip acquisition loop if not collecting
    while running:
        # poll for events
        save = False
        for event in pygame.event.get():
            # pygame.QUIT event means the user clicked X to close your window
            if event.type == pygame.QUIT:
                running = False

            # pygame.KEYDOWN event means the user pressed a key
            elif event.type == pygame.KEYDOWN:
                # save this frame to data when 's' is pressed
                if event.key == pygame.K_s:
                    save = True
                # exit when 'ESC' or 'q' is pressed
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        if save and acquisition is not None:
            if ckpt_path is not None:
                # recompute depth with foundation stereo
                acquisition = run_foundation_stereo(
                    acquisition, model, padders, metadata
                )

                # render the computed depth with detected markers
                rendering = render_acquisition(
                    acquisition, metadata, render_cfg, depth_render_cfg
                )

                pygame.surfarray.blit_array(screen, rendering.transpose(1, 0, 2))
                pygame.display.flip()
                time.sleep(pause_s)  # pause so the user can see the final acquisition

            acquisitions.append(acquisition)
            save_acquisition(acquisition, acquisitions_folder, len(acquisitions) - 1)

            if len(acquisitions) >= min_acquisitions:
                # update calibration
                intrinsics, extrinsics = calibrate(acquisitions, metadata)

            # reset the clock, so that the next interval starts from here
            clock.tick(fps)

        acquisition = {}

        # collect all images and robot state as close to concurrently as possible
        for camera in cameras:
            cam_name = camera.name
            acquisition[cam_name] = camera.get_observation()

        if "gripper_cam" in acquisition:
            ee_pos, xyzw = robot.get_ee_pose()
            # rearrange from xyzw to wxyz and convert to numpy
            wxyz = convert_quat(xyzw, to="wxyz")
            acquisition["gripper_cam"]["base_pose"] = torch.cat((ee_pos, wxyz)).numpy()

        acquisition = detect_markers(
            acquisition, charuco_detector, charuco_board, metadata
        )
        rendering = render_acquisition(
            acquisition, metadata, render_cfg, depth_render_cfg
        )

        pygame.surfarray.blit_array(screen, rendering.transpose(1, 0, 2))
        pygame.display.flip()
        clock.tick(fps)

    if use_pygame:
        log.debug("Quitting pygame...")
        pygame.quit()

    if do_render_fused_pcd or do_save_fused_pcd:
        pcd = render_fused_pointcloud(
            acquisitions[-1], intrinsics, extrinsics, metadata
        )
        if do_save_fused_pcd:
            log.info(f"Saving fused pointcloud to {fused_pcd_save_path}")
            o3d.t.io.write_point_cloud(fused_pcd_save_path, pcd)
        if do_render_fused_pcd:
            o3dvis.draw([pcd], title="Fused Pointcloud")

    # setup output folder
    if do_save_calibration:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        output_name = acquisitions_folder.name + ".yaml"
        output_path = output_folder / output_name
        log.info(f"Saving calibration to {output_path}")
        save_calibration(
            intrinsics,
            extrinsics,
            output_path,
            metadata,
            save_intrinsics=cfg.optimize_intrinsics,
            extrinsics_convention=cfg.extrinsics_convention,
        )

    if do_collect:
        log.debug("Stopping robot control...")
        robot.terminate_current_policy()
        del robot  # TODO: replace this with robot.close() when implemented

        log.debug("Closing cameras...")
        for camera in cameras:
            camera.close()

    log.debug("Program stopping...")


def detect_markers(
    acquisition: AcquisitionType,
    detector: aruco.CharucoDetector,
    board: aruco.CharucoBoard,
    metadata: MetadataType,
):
    for cam_name, cam_metadata in metadata.items():
        cam_acquisition = acquisition[cam_name]
        img = cam_acquisition[cam_metadata["stream_name"]]
        # Detect charuco corners (deterministic result)
        # The input image can be (H, W, 3) or (H, W)
        # corner_uvs: (M, 1, 2)
        corner_uvs, corner_ids, marker_corners, marker_ids = detector.detectBoard(img)
        if corner_uvs is not None:
            assert corner_ids is not None
            assert corner_uvs.shape[0] > 0 and corner_ids.shape[0] > 0
            assert corner_uvs.shape[0] == corner_ids.shape[0]
            assert corner_uvs.shape[1] == 1 and corner_ids.shape[1] == 1

            # obj_points: (M, 1, 3)
            # img_points: (M, 1, 2)
            obj_points, img_points = board.matchImagePoints(corner_uvs, corner_ids)
            assert np.allclose(img_points, corner_uvs)
        else:
            corner_uvs = np.zeros((0, 1, 2))
            corner_ids = np.zeros((0, 1))
            obj_points = np.zeros((0, 1, 3))

        acquisition[cam_name]["corner_image_uv"] = corner_uvs
        acquisition[cam_name]["corner_ids"] = corner_ids
        acquisition[cam_name]["corner_xyz_obj"] = obj_points

    return acquisition


def render_acquisition(
    acquisition: AcquisitionType,
    metadata: MetadataType,
    render_cfg: dict,
    depth_render_cfg: dict,
):
    height_width = (render_cfg["height"], render_cfg["width"])
    n_height, n_width = (render_cfg["n_height"], render_cfg["n_width"])

    renderings = []
    for cam_name, cam_metadata in metadata.items():
        cam_acquisition = acquisition[cam_name]
        img = cam_acquisition[cam_metadata["stream_name"]]
        # Render the image
        if img.ndim == 3:
            img_render = rgb_to_renderable(img.copy(), channel_order="HWC")
        elif img.ndim == 2:
            img_render = intensity_to_renderable(img.copy())
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        if len(cam_acquisition["corner_image_uv"]) > 0:
            aruco.drawDetectedCornersCharuco(
                img_render,
                cam_acquisition["corner_image_uv"],
                cam_acquisition["corner_ids"],
            )

        img_render = downsize_and_center_crop(img_render, height_width)
        renderings.append(img_render)

        depth = acquisition[cam_name][cam_metadata["depth_stream_name"]]
        depth_render = depth_to_renderable(depth, **depth_render_cfg)

        if (
            len(cam_acquisition["corner_image_uv"]) > 0
            and depth.shape == img.shape[:2]  # (H, W) vs. (H, W, C)
        ):
            aruco.drawDetectedCornersCharuco(
                depth_render,
                cam_acquisition["corner_image_uv"],
                cam_acquisition["corner_ids"],
            )

        depth_render = downsize_and_center_crop(depth_render, height_width)
        renderings.append(depth_render)

    rendering = tile_images(renderings, n_height, n_width, vertical=True)

    return rendering


def downsize_and_center_crop(image: np.ndarray, shape: tuple[int, int]):
    """Downsize an image to a desired target shape and retain aspect ratio by
    first scaling it down and then applying a center crop.
    """
    target_height, target_width = shape
    height, width = image.shape[:2]

    # Resize to make *both* dimensions ≥ target
    scale = max(target_height / height, target_width / width)
    temp_height, temp_width = int(height * scale), int(width * scale)

    # Resize without stretching
    image = cv2.resize(image, (temp_width, temp_height), interpolation=cv2.INTER_LINEAR)

    if image.shape[:2] != shape:
        # center crop to target size
        start_x = (temp_width - target_width) // 2
        start_y = (temp_height - target_height) // 2
        image = image[
            start_y : start_y + target_height, start_x : start_x + target_width
        ]

    return image


def load_foundation_stereo(
    ckpt_path: str, metadata: MetadataType
) -> tuple[
    nn.Module, dict[str, "third_party.FoundationStereo.core.utils.utils.InputPadder"]
]:
    for cam_name, cam_metadata in metadata.items():
        if cam_metadata["stream_name"] != "left":
            raise ValueError(
                f"FoundationStereo requires calibration of the left stream of each camera."
            )

        if "baseline" not in cam_metadata["intrinsics"]:
            raise ValueError(
                f"Camera {cam_name} is missing baseline in its intrinsics, which is required for FoundationStereo"
            )

    from third_party.FoundationStereo.core.foundation_stereo import FoundationStereo
    from third_party.FoundationStereo.core.utils.utils import InputPadder

    log.info(f"Loading FoundationStereo model from {ckpt_path}")

    cfg = OmegaConf.load(f"{os.path.dirname(ckpt_path)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"

    model = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.requires_grad_(False)
    model.cuda()
    model.eval()

    padders = {
        cam_name: InputPadder(
            tuple(cam_metadata["height_width"]), divis_by=32, force_square=False
        )
        for cam_name, cam_metadata in metadata.items()
    }

    return model, padders


def run_foundation_stereo(
    acquisition: AcquisitionType,
    model: nn.Module,
    padders: dict[str, "third_party.FoundationStereo.core.utils.utils.InputPadder"],
    metadata: MetadataType,
) -> AcquisitionType:
    # TODO: add option to use calibrated intrinsics
    valid_iters = model.args.valid_iters

    with torch.no_grad():
        for cam_name, cam_metadata in metadata.items():
            cam_acquisition = acquisition[cam_name]
            left, right = cam_acquisition["left"], cam_acquisition["right"]
            assert left.shape == right.shape
            # convert to float but leave in interval [0, 255]
            # add batch dimension
            left = torch.from_numpy(left).cuda().float().unsqueeze(dim=0)
            right = torch.from_numpy(right).cuda().float().unsqueeze(dim=0)
            if left.ndim == 3:
                # for intensity images, add channel dimension and repeat across all channels
                left = left.unsqueeze(-3).expand(-1, 3, -1, -1)
                right = right.unsqueeze(-3).expand(-1, 3, -1, -1)
            elif left.ndim == 4:
                # for rgb images, move channels to the front (in torch order)
                left = torch.movedim(left, -1, -3)
                right = torch.movedim(right, -1, -3)

            padder = padders[cam_name]
            left, right = padder.pad(left, right)

            disp = model.forward(left, right, iters=valid_iters, test_mode=True)

            disp = padder.unpad(disp)
            intrinsics = cam_metadata["intrinsics"]
            depth = intrinsics["fx"] * intrinsics["baseline"] / disp
            depth = depth.squeeze(dim=1).squeeze(dim=0).cpu().numpy()

            cam_acquisition[cam_metadata["depth_stream_name"]] = depth

    return acquisition


def render_fused_pointcloud(
    acquisition: AcquisitionType,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    metadata: MetadataType,
) -> o3d.geometry.Geometry:
    geometries = {}

    intrinsics = torch.from_numpy(intrinsics)
    extrinsics = torch.from_numpy(extrinsics)

    for i, (cam_name, cam_metadata) in enumerate(metadata.items()):
        cam_acquisition = acquisition[cam_name]
        img = cam_acquisition[cam_metadata["stream_name"]]
        depth = cam_acquisition[cam_metadata["depth_stream_name"]]
        cam_intrinsics = intrinsics[i]
        cam_extrinsics = extrinsics[i]

        depth = torch.from_numpy(depth)

        if "base_pose" in cam_acquisition:
            base_pose = torch.from_numpy(cam_acquisition["base_pose"])
            trans = base_pose[:3]
            rot = quaternion_to_matrix(base_pose[3:7])
            T_base2ee = make_pose(trans, rot)
            cam_extrinsics = T_base2ee @ cam_extrinsics

        # points: (H, W, 3)
        points = unproject_depth(depth, cam_intrinsics)
        points = transform_pointmap(points, cam_extrinsics).numpy()

        if img.ndim == 3:
            img_render = rgb_to_renderable(img, channel_order="HWC")
        elif img.ndim == 2:
            img_render = intensity_to_renderable(img)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        assert points.shape == img_render.shape
        points = points.reshape(-1, 3)
        img_render = img_render.reshape(-1, 3).astype(np.float32) / 255.0

        pcd = o3d.t.geometry.PointCloud()
        pcd.point.positions = o3d.core.Tensor.from_numpy(points)
        pcd.point.colors = o3d.core.Tensor.from_numpy(img_render)

        geometries[cam_name] = pcd

    geometries = list(geometries.values())
    fused = geometries[0]
    for geom in geometries[1:]:
        fused += geom

    return fused


def save_calibration(
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    output_path: Path,
    metadata: MetadataType,
    save_intrinsics: bool = True,
    extrinsics_convention: Literal["ros", "world"] = "ros",
):
    # the origin convention is ROS because we convert image points to camera
    # points by unprojecting depth maps, which gives points in ROS convention
    extrinsics = convert_extrinsics_convention(
        extrinsics, origin="ros", target=extrinsics_convention
    )

    conf = {}
    for i, cam_name in enumerate(metadata):
        conf[cam_name] = {}
        conf[cam_name]["extrinsics"] = extrinsics[i].tolist()
        if save_intrinsics:
            conf[cam_name]["intrinsics"] = intrinsics[i].tolist()

    conf = OmegaConf.create(conf)
    OmegaConf.save(conf, output_path)


if __name__ == "__main__":
    main()
