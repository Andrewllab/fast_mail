from abc import ABC, abstractmethod
from typing import Any, Mapping

from torch import Tensor

from environments.specs import CameraSpec


class BaseCamera(ABC):
    _name: str
    _height_width: tuple[int, int]
    _intrinsics: Mapping[str, Any] | None
    _extrinsics: Tensor | None
    _spec: CameraSpec

    @property
    def name(self) -> str:
        return self._name

    @property
    def height_width(self) -> tuple[int, int]:
        """
        Returns the height and width of the camera image.
        """
        return self._height_width

    @property
    def intrinsics(self) -> dict[str, Any]:
        if self._intrinsics is None:
            # if intrinsics are not provided, get them from the camera
            self._intrinsics = self.get_intrinsics()

        return dict(self._intrinsics)

    @property
    def extrinsics(self) -> Tensor | None:
        return self._extrinsics

    @property
    def spec(self) -> CameraSpec:
        return self._spec

    def get_intrinsics(self, stream_name: str = "default") -> dict[str, Any]:
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
