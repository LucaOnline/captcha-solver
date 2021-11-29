from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from data import build_dataset, Mode
from models import create_mask_model


def main():
    ds = build_dataset(Mode.Masks)
    val_ds = build_dataset(Mode.Masks)

    model = create_mask_model(150 * 300)

    cb = [
        EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=1),
        ModelCheckpoint("masks.hdf5", monitor="val_loss",
                        save_best_only=True, verbose=1),
        TensorBoard(),
    ]

    model.fit(ds, validation_data=val_ds, epochs=10, callbacks=cb, verbose=1)


if __name__ == "__main__":
    main()
