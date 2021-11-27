from data import build_dataset, Mode
from models import create_mask_segmentation_model


def main():
    ds = build_dataset(Mode.MaskSegments)
    model = create_mask_segmentation_model()


if __name__ == "__main__":
    main()
