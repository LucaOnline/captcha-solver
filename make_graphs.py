import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as area_under_curve
from tensorflow.keras.models import load_model
from xcaptcha.defaults import CHARSET_ALPHANUMERIC

from data import build_dataset, Mode


def sample_mask_predictions():
    dims = (75, 150)

    ds = build_dataset(dims, 1, Mode.Masks)
    model = load_model("masks.hdf5")

    img, mask = list(ds.batch(1).take(1).as_numpy_iterator())[0]
    predicted_mask = model.predict(img)

    img = np.reshape(img, dims)
    mask = np.reshape(mask, dims)
    predicted_mask = np.reshape(predicted_mask, dims)

    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.imshow("Mask", mask)
    cv.waitKey(0)
    cv.imshow("Predicted Mask", predicted_mask)
    cv.waitKey(0)

    cv.imwrite("out/masks_image.png", img)
    cv.imwrite("out/masks_label.png", mask)
    cv.imwrite("out/masks_pred.png", predicted_mask)


def make_roc_curve():
    ds = build_dataset((75, 150), 1000, Mode.Characters)
    model = load_model("characters.hdf5")

    binary_sentinel = "m"

    y_test = []
    y_scores = []
    for i, (mask, label) in ds.batch(1).enumerate():
        label_char = CHARSET_ALPHANUMERIC[np.argmax(label)]

        scores = model.predict(mask)[0]
        score = scores[np.argmax(scores)]

        # Manually build label arrays for sklearn since we're forcing a multiclass classifer
        # into its ROC curve functions
        if label_char == binary_sentinel:
            y_test.append(1)
        else:
            y_test.append(0)
        y_scores.append(score)

        predicted_char = CHARSET_ALPHANUMERIC[np.argmax(scores)]
        print(f"{i}: {(label_char, predicted_char)}")

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = area_under_curve(fpr, tpr)

    plt.plot(fpr, tpr, "b", label=f"AUC = {auc}")
    plt.legend()
    plt.plot([0, 1], [0, 1], "r--")  # Dashed line
    plt.xlim([0, 1])
    plt.xlabel("FPR")
    plt.ylim([0, 1])
    plt.ylabel("TPR")
    plt.title(f'ROC Curve (Positive class = "m", N = {len(y_test)})')
    plt.show()


if __name__ == "__main__":
    sample_mask_predictions()
