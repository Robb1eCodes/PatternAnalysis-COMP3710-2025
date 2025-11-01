import os
import tensorflow as tf

from dataset import load_data_2D

BASE = os.path.dirname(__file__)
TRAINED_MODEL = os.path.join(BASE, "best_trained_unet.h5")
TEST_IMAGES = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_test')
TEST_MASKS = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_seg_test')

TARGET_IMG_SHAPE = (128, 128)
NUM_CLASSES = 6

def main():

    if not os.path.exists(TRAINED_MODEL):
        raise FileNotFoundError(f"Model file not found at: {TRAINED_MODEL}")

    # Collect test image and mask file paths
    test_image_files = [os.path.join(TEST_IMAGES, f) for f in os.listdir(TEST_IMAGES) if f.endswith('.nii') or f.endswith('.nii.gz')]
    test_mask_files = [os.path.join(TEST_MASKS, f) for f in os.listdir(TEST_MASKS) if f.endswith('.nii') or f.endswith('.nii.gz')]
    X_test = load_data_2D(test_image_files, target_shape=TARGET_IMG_SHAPE)
    Y_test = load_data_2D(test_mask_files, target_shape=TARGET_IMG_SHAPE, categorical=True, num_classes=NUM_CLASSES)

    # load model
    model = tf.keras.models.load_model(TRAINED_MODEL, compile=False)

    # predict Y_test using X_test
    preds = model.predict(X_test, batch_size=8)