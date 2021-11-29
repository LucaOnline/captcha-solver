from data import build_dataset, Mode
from models import create_mask_segmentation_model


def main():
    dims = (150, 300)

    ds = build_dataset(dims, Mode.MaskSegments)
    model = create_mask_segmentation_model()


if __name__ == "__main__":
    main()
