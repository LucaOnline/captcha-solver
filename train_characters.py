from data import build_dataset, Mode
from models import create_characters_model


def main():
    ds = build_dataset(Mode.Characters)
    model = create_characters_model()


if __name__ == "__main__":
    main()
