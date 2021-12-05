from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from data import build_dataset, Mode
from models import create_mask_model
from options import IMAGE_DIMENSIONS, MASKS_MODEL_FILE


def main():
    ds = build_dataset(IMAGE_DIMENSIONS, 100, Mode.Masks).batch(10)
    val_ds = build_dataset(IMAGE_DIMENSIONS, 50, Mode.Masks).batch(10)

    model = create_mask_model(IMAGE_DIMENSIONS)

    monitor = "val_loss"
    cb = [
        EarlyStopping(monitor=monitor, mode="min", patience=10, verbose=1),
        ModelCheckpoint(MASKS_MODEL_FILE, monitor=monitor,
                        save_best_only=True, verbose=1),
        TensorBoard(),
    ]

    model.fit(ds, validation_data=val_ds, epochs=30, callbacks=cb, verbose=1)


if __name__ == "__main__":
    main()
