from typing import Tuple
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_mask_model(dims: Tuple[int, int]) -> Sequential:
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=dims+(1,)),
        Conv2D(32, (3, 3), activation="relu"),
        Conv2D(64, (1, 1), activation="relu"),
        MaxPooling2D(),

        Conv2D(32, (3, 3), activation="relu"),
        Conv2D(32, (3, 3), activation="relu"),
        Conv2D(64, (1, 1), activation="relu"),
        MaxPooling2D(),

        Conv2D(32, (3, 3), activation="relu"),
        Conv2D(32, (3, 3), activation="relu"),
        Conv2D(64, (1, 1), activation="relu"),
        MaxPooling2D(),

        Conv2D(64, (1, 1), activation="relu"),
        MaxPooling2D(),

        Flatten(),
        Dense(dims[0] * dims[1]),
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    return model


def create_mask_segmentation_model() -> Sequential:
    model = Sequential()
    return model


def create_characters_model() -> Sequential:
    model = Sequential()
    return model
