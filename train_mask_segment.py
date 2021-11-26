from .data import create_captcha_generator
from .models import create_mask_segmentation_model


def main():
    generator = create_captcha_generator()
    model = create_mask_segmentation_model()


if __name__ == "__main__":
    main()
