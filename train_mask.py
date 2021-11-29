from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from data import build_dataset, Mode
from models import create_mask_model


def main():
    dims = (100, 200)

    ds = build_dataset(dims, Mode.Masks).batch(10)
    val_ds = build_dataset(dims, Mode.Masks).batch(10)

    model = create_mask_model(dims)

    cb = [
        EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=1),
        ModelCheckpoint("masks.hdf5", monitor="val_loss",
                        save_best_only=True, verbose=1),
        TensorBoard(),
    ]

    model.fit(ds, validation_data=val_ds, epochs=10, callbacks=cb, verbose=1)


if __name__ == "__main__":
    main()
