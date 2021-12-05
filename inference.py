import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from xcaptcha.defaults import CHARSET_ALPHANUMERIC, FONTS
from xcaptcha.generator import CAPTCHAGenerator

from options import IMAGE_DIMENSIONS, MASKS_MODEL_FILE, MASK_SEGMENTS_MODEL_FILE, CHARS_MODEL_FILE, N_CHARACTERS

# Load the three models
masks_model = load_model(MASKS_MODEL_FILE)
mask_segments_model = load_model(MASK_SEGMENTS_MODEL_FILE)
chars_model = load_model(CHARS_MODEL_FILE)

# Generate our input data
generator = CAPTCHAGenerator(
    CHARSET_ALPHANUMERIC, IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, N_CHARACTERS, N_CHARACTERS, FONTS)

captcha = next(generator)

# Convert the image to greyscale
img = cv.cvtColor(captcha.image, cv.COLOR_BGR2GRAY)

# The input shape of the models
model_input_dims = (1, IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1], 1)

# Inferencing
mask_pred = masks_model.predict(np.reshape(img, model_input_dims))

mask_segments_pred = mask_segments_model.predict(
    np.reshape(mask_pred, model_input_dims))
mask_segments_pred = np.reshape(np.ceil(mask_segments_pred), IMAGE_DIMENSIONS)

# Rescale the predicted mask segments into our expected input range
mask_segments_pred = np.floor(np.abs((mask_segments_pred -
                                      np.mean(mask_segments_pred)) / np.std(mask_segments_pred)) * (N_CHARACTERS + 1))

# Predict each character separately using its mask
prediction = ""
for i in range(1, N_CHARACTERS + 1):
    char_mask = np.where(mask_segments_pred == i, np.ones(
        IMAGE_DIMENSIONS), np.zeros(IMAGE_DIMENSIONS))
    char_scores = chars_model.predict(np.reshape(char_mask, model_input_dims))
    char_pred = CHARSET_ALPHANUMERIC[np.argmax(char_scores)]

    prediction += char_pred

# Output final prediction
print(f"Predicted: {prediction} - Actual: {''.join(captcha.solution)}")
