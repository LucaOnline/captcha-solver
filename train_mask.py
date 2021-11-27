from data import build_dataset, Mode
from models import create_mask_model


def main():
    ds = build_dataset(Mode.Masks)
    print(ds.take(1))
    model = create_mask_model()


if __name__ == "__main__":
    main()
