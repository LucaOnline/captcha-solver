import cv2 as cv
import json
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from sklearn.metrics import roc_curve, auc as area_under_curve
from tensorflow.keras.models import load_model
from xcaptcha.defaults import CHARSET_ALPHANUMERIC

from data import build_dataset, Mode
from model_postprocessing import discretize_mask_segments_prediction, threshold_masks
from options import IMAGE_DIMENSIONS, MASKS_MODEL_FILE, MASK_SEGMENTS_MODEL_FILE, CHARS_MODEL_FILE, N_CHARACTERS


def sample_mask_predictions():
    ds = build_dataset(IMAGE_DIMENSIONS, 1, Mode.Masks)
    model = load_model(MASKS_MODEL_FILE)

    img, mask = list(ds.batch(1).take(1).as_numpy_iterator())[0]
    predicted_mask = model.predict(img)

    img = np.reshape(img, IMAGE_DIMENSIONS)
    mask = np.reshape(mask, IMAGE_DIMENSIONS)
    predicted_mask = np.reshape(predicted_mask, IMAGE_DIMENSIONS)
    predicted_mask = threshold_masks(predicted_mask)

    print(img)
    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.imshow("Mask", mask)
    cv.waitKey(0)
    cv.imshow("Predicted Mask", predicted_mask)
    cv.waitKey(0)

    cv.imwrite("out/masks_image.png", img * 255)
    cv.imwrite("out/masks_label.png", mask * 255)
    cv.imwrite("out/masks_pred.png", predicted_mask * 255)


def random_color() -> tuple:
    return (randint(0, 255), randint(0, 255), randint(0, 255))


def colorize_mask_segments(input: np.ndarray, n_segments: int) -> np.ndarray:
    h, w = input.shape
    output = np.zeros(input.shape + (3,), dtype=np.uint8)
    colors = [random_color() for _ in range(n_segments)]
    for n in range(1, n_segments + 1):
        for r in range(h):
            for c in range(w):
                if input[r, c] == n:
                    output[r, c] = colors[n - 1]

    return output


def sample_mask_segment_predictions():
    ds = build_dataset(IMAGE_DIMENSIONS, 1, Mode.MaskSegments)
    model = load_model(MASK_SEGMENTS_MODEL_FILE)

    mask, mask_segments = list(ds.batch(1).take(1).as_numpy_iterator())[0]
    predicted_mask_segments = model.predict(mask)
    predicted_mask_segments = discretize_mask_segments_prediction(
        predicted_mask_segments, N_CHARACTERS)

    mask = np.reshape(mask, IMAGE_DIMENSIONS)

    # Give each mask segment a unique color
    mask_segments = colorize_mask_segments(
        np.reshape(mask_segments, IMAGE_DIMENSIONS), 5)
    predicted_mask_segments = colorize_mask_segments(
        np.reshape(predicted_mask_segments, IMAGE_DIMENSIONS), 5)

    cv.imshow("Mask", mask)
    cv.waitKey(0)
    cv.imshow("Mask Segments", mask_segments)
    cv.waitKey(0)
    cv.imshow("Predicted Mask Segments", predicted_mask_segments)
    cv.waitKey(0)

    cv.imwrite("out/mask_segments_mask.png", mask * 255)
    cv.imwrite("out/mask_segments_label.png", mask_segments)
    cv.imwrite("out/mask_segments_pred.png", predicted_mask_segments)


def make_roc_curve():
    ds = build_dataset(IMAGE_DIMENSIONS, 1000, Mode.Characters)
    model = load_model(CHARS_MODEL_FILE)

    binary_sentinel = "X"

    total_inferences = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    y_test = []
    y_scores = []
    for i, (mask, label) in ds.batch(1).enumerate():
        label_char = CHARSET_ALPHANUMERIC[np.argmax(label)]

        scores = model.predict(mask)[0]
        score = scores[np.argmax(scores)]
        pred_char = CHARSET_ALPHANUMERIC[np.argmax(scores)]

        # Update confusion matrix data
        total_inferences += 1
        if label_char == binary_sentinel and pred_char == binary_sentinel:
            tp += 1
        elif label_char != binary_sentinel and pred_char == binary_sentinel:
            fp += 1
        elif label_char != binary_sentinel and pred_char != binary_sentinel:
            tn += 1
        elif label_char == binary_sentinel and pred_char != binary_sentinel:
            fn += 1

        # Manually build label arrays for sklearn since we're forcing a multiclass classifer
        # into its ROC curve functions
        if label_char == binary_sentinel:
            y_test.append(1)
        else:
            y_test.append(0)
        y_scores.append(score)

        predicted_char = CHARSET_ALPHANUMERIC[np.argmax(scores)]
        print(f"{i}: {(label_char, predicted_char)}")

    # Output confusion matrix data
    confusion = {
        "total_inferences": total_inferences,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }

    with open("out/confusion_matrix.json", "w+") as f:
        f.write(json.dumps(confusion))

    # Produce ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = area_under_curve(fpr, tpr)

    plt.plot(fpr, tpr, "b", label=f"AUC = {auc}")
    plt.legend()
    plt.plot([0, 1], [0, 1], "r--")  # Dashed line
    plt.xlim([0, 1])
    plt.xlabel("FPR")
    plt.ylim([0, 1])
    plt.ylabel("TPR")
    plt.title(
        f'ROC Curve (Positive class = "{binary_sentinel}", N = {len(y_test)})')
    plt.show()


if __name__ == "__main__":
    sample_mask_predictions()
    sample_mask_segment_predictions()
    make_roc_curve()
