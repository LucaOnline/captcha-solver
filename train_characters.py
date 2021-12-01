from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from xcaptcha.defaults import CHARSET_ALPHANUMERIC

from data import build_dataset, Mode
from models import create_characters_model


def main():
    dims = (75, 150)

    ds = build_dataset(dims, 10000, Mode.Characters).batch(50)
    val_ds = build_dataset(dims, 2000, Mode.Characters).batch(50)

    model = create_characters_model(dims, len(CHARSET_ALPHANUMERIC))

    monitor = "loss"
    cb = [
        EarlyStopping(monitor=monitor, mode="min", patience=10, verbose=1),
        ModelCheckpoint("characters.hdf5", monitor=monitor,
                        save_best_only=True, verbose=1),
        TensorBoard(),
    ]

    model.fit(ds, validation_data=val_ds, epochs=50, callbacks=cb, verbose=1)


if __name__ == "__main__":
    main()
