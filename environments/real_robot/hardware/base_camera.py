from abc import ABC, abstractmethod
from typing import Any

from environments.specs import CameraSpec


class BaseCamera(ABC):
    _name: str
    _spec: CameraSpec

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> CameraSpec:
        return self._spec

    @property
    def height_width(self) -> tuple[int, int]:
        """
        Returns the height and width of the camera image.
        """
        raise NotImplementedError

    def get_intrinsics(self, stream_name: str) -> dict[str, Any]:
        """
        Returns the camera intrinsics.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        """
        Closes the camera connection.
        """
        pass
