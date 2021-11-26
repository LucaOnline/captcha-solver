from xcaptcha.defaults import CHARSET_ALPHANUMERIC, FONTS
from xcaptcha.generator import CAPTCHAGenerator


def create_captcha_generator() -> CAPTCHAGenerator:
    generator = CAPTCHAGenerator(
        CHARSET_ALPHANUMERIC, (150, 300), (200, 400), 5, 7, FONTS)
    return generator
