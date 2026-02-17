import logging
import multiprocessing as mp
import traceback

import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.visualization as o3dvis
import torch.utils.dlpack

ctx = mp.get_context("spawn")
log = logging.getLogger(__name__)


def torch_to_o3d(tensor: torch.Tensor) -> o3c.Tensor:
    """
    Convert a pytorch tensor (cpu or cuda) to an Open3D tensor.
    """
    return o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def numpy_to_o3d(array: np.ndarray) -> o3c.Tensor:
    """
    Convert a numpy array to an Open3D tensor.
    """
    return o3c.Tensor.from_numpy(array)


def o3d_to_torch(o3d_tensor: o3c.Tensor) -> torch.Tensor:
    """
    Convert an Open3D tensor to a pytorch tensor.
    """
    return torch.utils.dlpack.from_dlpack(o3d_tensor.to_dlpack())


class AsyncPcdRenderer(ctx.Process):
    def __init__(
        self,
        width: int = 1024,
        height: int = 768,
        window_name: str = "PointCloud",
        coordinate_frame_size: float = 0.1,
        log_pointcloud_size: bool = False,
        show_origin: bool = True,
    ) -> None:

        super().__init__(daemon=True)

        self.width = width
        self.height = height
        self.window_name = window_name
        self.coordinate_frame_size = coordinate_frame_size
        self.log_pointcloud_size = log_pointcloud_size
        self.show_origin = show_origin

        self.child_pipe, self.parent_pipe = ctx.Pipe(duplex=False)
        self.start()

    def render_pcd(
        self, points: np.ndarray, colors: np.ndarray | None, name: str = "pcd"
    ) -> None:
        if self.log_pointcloud_size:
            log.info(f"Rendering point cloud '{name}' with {points.shape[0]} points.")

        self._send_to_child(
            (
                "update_geometry",
                {
                    "type": "PointCloud",
                    "name": name,
                    "points": points,
                    "color": colors,
                },
            )
        )

    def render_coordinate_frame(
        self,
        translation: np.ndarray,
        rotation: np.ndarray,
        name: str,
        size: float | None = None,
    ) -> None:
        self._send_to_child(
            (
                "update_geometry",
                {
                    "type": "CoordinateFrame",
                    "name": name,
                    "translation": translation,
                    "rotation": rotation,
                    "size": size or self.coordinate_frame_size,
                },
            )
        )

    def _send_to_child(self, msg: object) -> None:
        try:
            self.parent_pipe.send(msg)
        except (BrokenPipeError, EOFError):
            # rendering process has crashed or exited
            raise KeyboardInterrupt("Open3D visualizer quit")

    def run(self) -> None:

        vis = o3dvis.Visualizer()
        vis.create_window(self.window_name, self.width, self.height)
        geometries = {}

        if self.show_origin:
            # add an extra large coordinate frame at the origin
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.coordinate_frame_size * 3, origin=[0, 0, 0]
            )
            vis.add_geometry(origin)
            geometries["origin"] = origin

        try:
            while True:
                # Non-blocking check for incoming messages
                if self.child_pipe.poll():
                    cmd, data = self.child_pipe.recv()
                    if cmd == "QUIT":
                        break

                    elif cmd == "update_geometry":
                        match data["type"]:
                            case "PointCloud":
                                name = data["name"]
                                points = data["points"]
                                points = o3d.utility.Vector3dVector(points)
                                color = data["color"]
                                if color is not None:
                                    if color.dtype == np.uint8:
                                        color = color.astype(np.float32) / 255.0
                                    color = o3d.utility.Vector3dVector(color)

                                if name in geometries:
                                    pcd = geometries[name]
                                    pcd.points = points
                                    if color is not None:
                                        pcd.colors = color
                                    vis.update_geometry(pcd)

                                else:
                                    # on first call
                                    pcd = o3d.geometry.PointCloud(points)
                                    if color is not None:
                                        pcd.colors = color
                                    vis.add_geometry(pcd)
                                    geometries[name] = pcd

                            case "CoordinateFrame":
                                name = data["name"]
                                translation = data["translation"]
                                rotation = data["rotation"]
                                size = data["size"]

                                # since we can't set an absolute pose, remove the old coordinate
                                # frame and add a new one with the correct pose
                                try:
                                    frame = geometries[name]
                                    vis.remove_geometry(frame, reset_bounding_box=False)
                                except KeyError:
                                    pass

                                frame = (
                                    o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=size,
                                        origin=translation,
                                    )
                                )
                                frame.rotate(rotation, center=translation)
                                vis.add_geometry(frame, reset_bounding_box=False)
                                geometries[name] = frame

                            case _:
                                print(f"Unknown geometry type: {data['type']}")

                    else:
                        print(f"Unknown command: {cmd}")

                # Keep the window responsive
                vis.poll_events()
                vis.update_renderer()

        except KeyboardInterrupt:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            try:
                vis.destroy_window()
            except Exception:
                pass

            self.child_pipe.close()

    def close(self) -> None:
        self._send_to_child(("QUIT", {}))
        self.terminate()
        self.join()
        self.parent_pipe.close()
