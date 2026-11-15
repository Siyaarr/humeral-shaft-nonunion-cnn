"""
Pre-downloads all pretrained model weights used by this project into the
local Keras cache (`%USERPROFILE%\\.keras\\models\\`).

Intended workflow:
  1. Run this on an online machine.
  2. Copy the entire `.keras\\models\\` folder to the offline machine,
     placing it at the same location under that user's profile.
  3. main.py will then load weights from cache instead of attempting to
     download them.
"""

import os
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet152, DenseNet201, EfficientNetB0, EfficientNetB3,
    MobileNetV3Large, InceptionV3, Xception, InceptionResNetV2,
)

MODELS = [
    (VGG16,             (224, 224, 3)),
    (VGG19,             (224, 224, 3)),
    (ResNet152,         (256, 256, 3)),
    (DenseNet201,       (256, 256, 3)),
    (EfficientNetB0,    (256, 256, 3)),
    (EfficientNetB3,    (256, 256, 3)),
    (MobileNetV3Large,  (256, 256, 3)),
    (InceptionV3,       (299, 299, 3)),
    (Xception,          (299, 299, 3)),
    (InceptionResNetV2, (299, 299, 3)),
]

def main():
    cache_dir = os.path.join(os.path.expanduser("~"), ".keras", "models")
    print(f"Target cache directory: {cache_dir}\n")

    for loader, shape in MODELS:
        name = loader.__name__
        print(f"[{name}] downloading (include_top=False, weights='imagenet')...")
        loader(include_top=False, weights="imagenet", input_shape=shape)
        print(f"[{name}] done.\n")

    print("All weights cached.")
    print(f"Copy this folder to the offline PC's same location:\n  {cache_dir}")

if __name__ == "__main__":
    main()
