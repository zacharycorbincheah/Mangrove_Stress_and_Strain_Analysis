"""Video / image-sequence frame loading."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


class FrameSource:
    """Yields greyscale float64 frames from a video file or a folder of images."""

    def __init__(self, path: str | Path, start: int = 0, stop: int | None = None, step: int = 1):
        self.path = Path(path)
        self.start, self.stop, self.step = start, stop, max(1, step)
        if self.path.is_dir():
            self.files = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self.files:
                raise FileNotFoundError(f"no images found in {self.path}")
            self.n_frames = len(self.files)
            self.fps = None
        else:
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise FileNotFoundError(f"cannot open video {self.path}")
            self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and fps > 0 else None
            cap.release()
            self.files = None

    def indices(self) -> list[int]:
        stop = self.n_frames if self.stop is None else min(self.stop, self.n_frames)
        return list(range(self.start, stop, self.step))

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame.astype(np.float64)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        wanted = self.indices()
        if self.files is not None:
            for i in wanted:
                img = cv2.imread(str(self.files[i]), cv2.IMREAD_UNCHANGED)
                yield i, self._to_gray(img)
            return
        wanted_set = set(wanted)
        last = wanted[-1] if wanted else -1
        cap = cv2.VideoCapture(str(self.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        i = self.start
        try:
            while i <= last:
                ok, frame = cap.read()
                if not ok:
                    break
                if i in wanted_set:
                    yield i, self._to_gray(frame)
                i += 1
        finally:
            cap.release()

    def read(self, index: int) -> np.ndarray:
        if self.files is not None:
            return self._to_gray(cv2.imread(str(self.files[index]), cv2.IMREAD_UNCHANGED))
        cap = cv2.VideoCapture(str(self.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise IndexError(f"cannot read frame {index}")
        return self._to_gray(frame)


def select_roi(image: np.ndarray) -> tuple[int, int, int, int]:
    """Interactive ROI selection window. Returns (x, y, w, h)."""
    disp = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    roi = cv2.selectROI("Select ROI, then press ENTER", disp, showCrosshair=True)
    cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        raise ValueError("empty ROI selected")
    return tuple(int(v) for v in roi)
