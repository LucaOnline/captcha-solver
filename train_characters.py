from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from xcaptcha.defaults import CHARSET_ALPHANUMERIC

from data import build_dataset, Mode
from models import create_characters_model
from options import IMAGE_DIMENSIONS, CHARS_MODEL_FILE


def main():
    ds = build_dataset(IMAGE_DIMENSIONS, 10000, Mode.Characters).batch(50)
    val_ds = build_dataset(IMAGE_DIMENSIONS, 2000, Mode.Characters).batch(50)
    test_ds = build_dataset(IMAGE_DIMENSIONS, 1000, Mode.Characters).batch(50)

    model = create_characters_model(
        IMAGE_DIMENSIONS, len(CHARSET_ALPHANUMERIC))

    monitor = "val_loss"
    cb = [
        EarlyStopping(monitor=monitor, mode="min", patience=10, verbose=1),
        ModelCheckpoint(CHARS_MODEL_FILE, monitor=monitor,
                        save_best_only=True, verbose=1),
        TensorBoard(),
    ]

    model.fit(ds, validation_data=val_ds, epochs=50, callbacks=cb, verbose=1)
    model.evaluate(test_ds, verbose=1)


if __name__ == "__main__":
    main()
