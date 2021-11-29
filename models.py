from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def create_mask_model(dims: Tuple[int, int]) -> Sequential:
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu",
               kernel_initializer="he_normal", input_shape=dims+(3,)),
        Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal"),
        MaxPooling2D(),

        Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal"),
        Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal"),
        MaxPooling2D(),

        Conv2D(64, (1, 1), activation="relu", kernel_initializer="he_normal"),
        MaxPooling2D(),

        Flatten(),
        Dense(dims[0] * dims[1], kernel_initializer="he_normal"),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


def create_mask_segmentation_model() -> Sequential:
    model = Sequential()
    return model


def create_characters_model() -> Sequential:
    model = Sequential()
    return model
