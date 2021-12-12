# captcha-solver
A 3-model image CAPTCHA solver.

## Setup
Python 3.8.3+ is required to run this project. To install the project dependencies, simply run `pip install -r requirements.txt` in the project directory.

## Training
Pre-trained models are included in this repository, so for inferencing no model training is required.

Each of the models may be trained separately, using the `train_*.py` files in the repository.

These models were trained on a single NVIDIA GeForce GTX 1650 with 4GB GDDR6 VRAM. Training on GPU requires following the additional setup instructions for GPU training
on the [Tensorflow](https://www.tensorflow.org/install/gpu) website. However, for the version of Tensorflow used in this project, use CUDA 10.1 and cuDNN SDK v7.6.5 for
CUDA 10.1.

To train each model, execute the following commands:
```sh
python ./train_mask.py
python ./train_mask_segments.py
python ./train_characters.py
```

## Prediction
No additional software is required for inferencing. To run the pipeline with a randomly-generated input, run the command `python ./inference.py`. The command output will
include the actual and predicted text sequences. The generated image will be saved to `out/inference_input.png`.

## Evaluation
Sample outputs for the 3 models can be generated at once by running `python ./make_graphs.py`. This will display several data images, which may be dismissed with a keyboard
press. The script will generate several files:
![Generated files](https://raw.githubusercontent.com/LucaOnline/captcha-solver/main/assets/0.png)
```
 - masks_image.png: A greyscale image used as input for the masks model.
 - masks_label.png: The ground-truth text mask for the input image.
 - masks_pred.png: The predicted mask.
 - mask_segments_mask.png: An input text mask used as input for the mask segmentation model.
 - mask_segments_label.png: The colorized ground-truth labels for the input data.
 - mask_segments_pred.png: The predicted mask segment labels.
 - confusion_matrix.json: Confusion matrix data for the characters model. The positive class is hardcoded to be "X".
```

Additionally, an ROC curve for the characters model will be graphed and displayed. This may be saved using the Save button in the display window, shown in the following screenshot.
In the above screenshot, it has been saved to `chars_roc.png`. The ROC curve and the area under the curve may differ from the following screenshot.
![Matplotlib display of characters model ROC curve](https://raw.githubusercontent.com/LucaOnline/captcha-solver/main/assets/2.png)

## Data
All data is generated at runtime using the `xcaptcha` library. Besides installing the dependencies described in `requirements.txt`, no additional data downloads are needed.
