import cv2
import numpy as np


class FrameReader:
    def __init__(self, source: str | int, target_fps: int = 20) -> None:
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open video source: {source}"
            )
        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._native_fps <= 0:
            self._native_fps = target_fps
        self._skip_interval = max(1, round(self._native_fps / target_fps))
        self._frame_index = 0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_width(self) -> int:
        return self.width

    @property
    def frame_height(self) -> int:
        return self.height

    @property
    def native_fps(self) -> float:
        return self._native_fps

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None, int]:
        while True:
            ret, frame = self._cap.read()
            if not ret:
                return False, None, self._frame_index
            self._frame_index += 1
            if (self._frame_index % self._skip_interval) == 0:
                return True, frame, self._frame_index

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *args) -> None:
        self.release()
