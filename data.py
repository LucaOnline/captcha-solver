import enum
import numpy as np
import tensorflow as tf
from xcaptcha.defaults import CHARSET_ALPHANUMERIC, FONTS
from xcaptcha.generator import CAPTCHAGenerator


class Mode(enum.Enum):
    Masks = 0
    MaskSegments = 1
    Characters = 2


class CAPTCHADatasetSource(CAPTCHAGenerator):
    def __init__(self, mode: tf.int32, dims: tuple):
        super().__init__(CHARSET_ALPHANUMERIC, dims, dims, 5, 7, FONTS)
        self.mode = Mode(int(mode))
        self.solution = ""
        self.solution_idx = -1
        self.solution_masks = {}

    def __next__(self):
        if self.mode == Mode.Characters and self.solution_idx != len(self.solution):
            # X=single character mask, Y=character
            next_char = self.solution[self.solution_idx]
            next_data = (tf.convert_to_tensor(
                self.solution_masks[next_char], np.float64), np.float32(ord(next_char)))
            self.solution_idx += 1
            return next_data

        # Generate next CAPTCHA
        captcha_info = super().__next__()

        masks = list(captcha_info.masks.values())
        if self.mode == Mode.Masks:
            # X=image, Y=merged masks
            return (tf.convert_to_tensor(captcha_info.image), tf.convert_to_tensor(self.merge_masks(np.array(masks))))
        elif self.mode == Mode.MaskSegments:
            # X=merged masks, Y=merged masks with distinct values

            # Merge masks, using a distinct value for each layer
            label_mask = np.copy(masks[0])
            for i in range(len(masks) - 1):
                bottom = label_mask
                top = masks[i + 1] * (i + 1)
                label_mask = np.where(top == i+1, top, bottom)

            return (tf.convert_to_tensor(self.merge_masks(np.array(masks))), tf.convert_to_tensor(label_mask))
        else:  # self.mode == Mode.Characters
            # Prepare CAPTCHA to be reused for each character in the image
            self.solution = captcha_info.solution
            self.solution_idx = 0
            self.solution_masks = captcha_info.masks
            return self()


def build_dataset(mode: Mode) -> tf.data.Dataset:
    return tf.data.Dataset.from_generator(CAPTCHADatasetSource, args=[np.int32(mode.value), (150, 300)], output_types=tf.float32, output_shapes=())
