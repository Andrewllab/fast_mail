import logging

import hydra
import numpy as np
import open3d as o3d
import open3d.visualization as o3dvis
import rootutils
from omegaconf import DictConfig
from tensordict import TensorDict

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from environments.datamodule import TrajectoryDataModule
from utils.conf import setup_resolvers
from utils.instantiators import instantiate_datamodule
from utils.logging import configure_logging
from utils.math import quaternion_to_matrix

log = logging.getLogger(__name__)

COLOR_MAPPING = {
    (1.0, 0.0, 0.0): (0.7, 0.0, 0.0),
    (0.0, 1.0, 0.0): (0.0, 0.7, 0.0),
    (0.0, 0.0, 1.0): (0.0, 0.0, 0.7),
    # (0.5, 0.5, 0.5): (0.5, 0.5, 0.5),
}
ARROW_SIZE = 0.03


@hydra.main(
    version_base=None, config_path="../configs", config_name="visualize_dataset"
)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.python_logging)

    # instantiate dataset
    datamodule: TrajectoryDataModule = instantiate_datamodule(cfg.data)

    # manually run prepare data and setup so we can use dataset specs for model creation
    log.debug("Instantiating datamodule...")
    datamodule.prepare_data()
    datamodule.setup(stage="predict")

    dataset = datamodule.eval_dataset
    trajectories = dataset.trajectories

    for traj in trajectories:
        visualize_trajectory(traj)


def visualize_trajectory(trajectory: TensorDict) -> None:

    ee_poses = trajectory["obs", "ee_pose"]
    target_ee_poses = trajectory["action"][..., :-1, :7]  # remove gripper width

    recolored = None

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=ARROW_SIZE * 5)

    geometries = [origin]
    for ee_pose, target_ee_pose in zip(ee_poses, target_ee_poses):

        # create a coordinate frame for each pose
        pos, rot = ee_pose[:3], ee_pose[3:7]
        rot = quaternion_to_matrix(rot)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=ARROW_SIZE, origin=pos
        )
        frame.rotate(rot, center=pos)

        # create a coordinate frame for each pose
        pos, rot = target_ee_pose[:3], target_ee_pose[3:7]
        rot = quaternion_to_matrix(rot)
        target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=ARROW_SIZE * 0.75, origin=pos
        )
        target_frame.rotate(rot, center=pos)

        if recolored is None:
            colors = np.asarray(frame.vertex_colors)
            recolored = recolor(colors, COLOR_MAPPING)
        target_frame.vertex_colors = o3d.utility.Vector3dVector(recolored)

        geometries.extend([frame, target_frame])

    o3dvis.draw_geometries(geometries)


def recolor(
    colors: np.ndarray,
    mapping: dict[tuple[float, float, float], tuple[float, float, float]],
) -> np.ndarray:
    """
    Recolor a point cloud based on a mapping of original colors to new colors.

    Args:
        colors: The original colors of the point cloud.
        mapping: A dictionary mapping original colors to new colors.

    Returns:
        The recolored point cloud.
    """

    recolored = colors.copy()
    for orig_color, new_color in mapping.items():
        mask = np.all(colors == orig_color, axis=1)
        recolored[mask] = new_color
    return recolored


if __name__ == "__main__":
    setup_resolvers()
    main()
