import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as area_under_curve
from tensorflow.keras.models import load_model
from xcaptcha.defaults import CHARSET_ALPHANUMERIC

from data import build_dataset, Mode


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
    make_roc_curve()
