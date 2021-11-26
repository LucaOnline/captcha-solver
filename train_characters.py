from .data import create_captcha_generator
from .models import create_characters_model


def main():
    generator = create_captcha_generator()
    model = create_characters_model()


if __name__ == "__main__":
    main()
