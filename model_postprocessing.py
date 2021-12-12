import numpy as np


def discretize_mask_segments_prediction(predicted_mask_segments: np.ndarray, n_chars: int) -> np.ndarray:
    return np.floor((predicted_mask_segments -
                     np.mean(predicted_mask_segments)) / np.std(predicted_mask_segments) * (n_chars + 1))


def threshold_masks(pred: np.ndarray) -> np.ndarray:
    return np.where(pred >= 0.5, np.ones(pred.shape), np.zeros(pred.shape))
