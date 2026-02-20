from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np


@dataclass
class BackgroundParams:
    fg_threshold: int = 25         # fixed threshold on absdiff grayscale
    blur_ksize: int = 5            # blur reduces sensor noise differences
    morph_ksize: int = 5           # clean specks
    dilate_iters: int = 1          # expand mask a bit
    use_otsu: bool = False         # set True only if you want auto-threshold


class BackgroundModel:
    def __init__(self, params: BackgroundParams = BackgroundParams()):
        self.params = params
        self._bg_rgb: Optional[np.ndarray] = None

    def is_ready(self) -> bool:
        return self._bg_rgb is not None

    def build_from_frames(self, frames_rgb: List[np.ndarray]) -> None:
        if not frames_rgb:
            raise ValueError("No frames provided for background")
        stack = np.stack(frames_rgb, axis=0).astype(np.uint8)
        med = np.median(stack, axis=0).astype(np.uint8)
        self._bg_rgb = med

    def get_background(self) -> Optional[np.ndarray]:
        return None if self._bg_rgb is None else self._bg_rgb.copy()

    def foreground_mask(self, current_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._bg_rgb is None:
            return None

        cur_g = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
        bg_g = cv2.cvtColor(self._bg_rgb, cv2.COLOR_RGB2GRAY)

        diff = cv2.absdiff(cur_g, bg_g)

        k = int(self.params.blur_ksize)
        if k > 1:
            if k % 2 == 0:
                k += 1
            diff = cv2.GaussianBlur(diff, (k, k), 0)

        if self.params.use_otsu:
            _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(diff, int(self.params.fg_threshold), 255, cv2.THRESH_BINARY)

        mk = int(self.params.morph_ksize)
        if mk > 1:
            if mk % 2 == 0:
                mk += 1
            kernel = np.ones((mk, mk), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if int(self.params.dilate_iters) > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=int(self.params.dilate_iters))

        return mask.astype(np.uint8)
