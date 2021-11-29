from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, BatchNormalization, Flatten, Dense


def create_mask_model(num_classes: int) -> Sequential:
    model = Sequential([
        Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        DepthwiseConv2D(16, (3, 3), padding="same",
                        kernel_initializer="he_normal"),
        BatchNormalization(),
        MaxPooling2D(),

        Flatten(),
        Dense(64, activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        Dense(num_classes, activation="softmax",
              kernel_initializer="he_normal"),
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
