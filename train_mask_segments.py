from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from data import build_dataset, Mode
from models import create_mask_segmentation_model


def main():
    dims = (75, 150)

    ds = build_dataset(dims, 100, Mode.MaskSegments).batch(10)
    val_ds = build_dataset(dims, 50, Mode.MaskSegments).batch(10)

    model = create_mask_segmentation_model(dims)

    monitor = "val_loss"
    cb = [
        EarlyStopping(monitor=monitor, mode="min", patience=10, verbose=1),
        ModelCheckpoint("mask_segments.hdf5", monitor=monitor,
                        save_best_only=True, verbose=1),
        TensorBoard(),
    ]

    model.fit(ds, validation_data=val_ds, epochs=30, callbacks=cb, verbose=1)


if __name__ == "__main__":
    main()
