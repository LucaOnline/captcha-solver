import cv2 as cv
import enum
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from xcaptcha.defaults import CHARSET_ALPHANUMERIC, FONTS
from xcaptcha.generator import CAPTCHAGenerator


class Mode(enum.Enum):
    Masks = 0
    MaskSegments = 1
    Characters = 2


class CAPTCHADatasetSource(CAPTCHAGenerator):
    def __init__(self, mode: tf.int32, n_data_points: int, dims: tuple):
        super().__init__(CHARSET_ALPHANUMERIC, dims, dims, 5, 5, FONTS)
        self.mode = Mode(int(mode))
        self.n_data_points = n_data_points
        self.i_data_point = 0
        self.solution = ""
        self.solution_idx = -1
        self.solution_masks = {}

    def __next__(self):
        if self.i_data_point == self.n_data_points:
            raise StopIteration()
        self.i_data_point += 1

        if self.mode == Mode.Characters and self.solution_idx != len(self.solution):
            # X=single character mask, Y=character
            next_char = self.solution[self.solution_idx]
            next_data = (tf.convert_to_tensor(
                self.solution_masks[next_char], np.float32), np.float32(ord(next_char)))
            self.solution_idx += 1
            return next_data

        # Generate next CAPTCHA
        captcha_info = super().__next__()

        masks = list(captcha_info.masks.values())
        if self.mode == Mode.Masks:
            # X=image, Y=merged masks
            return (self.format_image(captcha_info.image), self.masks_to_tensor(masks))
        elif self.mode == Mode.MaskSegments:
            # X=merged masks, Y=merged masks with distinct values
            label_mask = self.merge_masks_distinct(masks)
            return (self.masks_to_tensor(masks), tf.convert_to_tensor(label_mask))
        else:  # self.mode == Mode.Characters
            # Prepare CAPTCHA to be reused for each character in the image
            self.solution = captcha_info.solution
            self.solution_idx = 0
            self.solution_masks = captcha_info.masks
            return self()

    def merge_masks_distinct(self, masks: List[np.ndarray]) -> np.ndarray:
        # Merge masks, using a distinct value for each layer
        label_mask = np.copy(masks[0])
        for i in range(len(masks) - 1):
            bottom = label_mask
            top = masks[i + 1] * (i + 1)
            label_mask = np.where(top == i+1, top, bottom)
        return label_mask

    def format_image(self, image: np.ndarray) -> tf.Tensor:
        h, w, _ = image.shape
        return tf.convert_to_tensor(np.reshape(self.rescale_image(cv.cvtColor(image, cv.COLOR_BGR2GRAY)), (h, w, 1)))

    def rescale_image(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255

    def masks_to_tensor(self, masks: List[np.ndarray]) -> tf.Tensor:
        h, w = masks[0].shape
        return tf.convert_to_tensor(np.reshape(self.merge_masks(np.array(masks)), (h * w,)))


def build_dataset(dims: Tuple[int, int], n_data_points: int, mode: Mode) -> tf.data.Dataset:
    return tf.data.Dataset.from_generator(
        CAPTCHADatasetSource,
        args=[np.int32(mode.value), n_data_points, dims],
        output_types=(tf.float32, tf.float32),
        output_shapes=(dims + (1,), dims[0] * dims[1]))
