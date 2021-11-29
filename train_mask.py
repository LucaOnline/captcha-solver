from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from data import build_dataset, Mode
from models import create_mask_model


def main():
    dims = (75, 150)

    ds = build_dataset(dims, Mode.Masks).batch(1)

    model = create_mask_model(dims)

    monitor = "loss"
    cb = [
        EarlyStopping(monitor=monitor, mode="min", patience=10, verbose=1),
        ModelCheckpoint("masks.hdf5", monitor=monitor,
                        save_best_only=True, verbose=1),
        TensorBoard(),
    ]

    model.fit(ds, epochs=10, callbacks=cb, verbose=1)


if __name__ == "__main__":
    main()
