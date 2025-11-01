import os

from dataset import load_data_2D
from modules import unet_2d

BASE = os.path.dirname(__file__)
TRAIN_IMAGES = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_train')
TRAIN_MASKS = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_seg_train')
VAL_IMAGES = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_validate')
VAL_MASKS = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate')

TARGET_IMG_SHAPE = (128, 128)

def main():
    # base file path
    base = os.path.dirname(__file__)

    # 
    print('Loading train data...')
    train_image_files = [os.path.join(TRAIN_IMAGES, f) for f in os.listdir(TRAIN_IMAGES) if f.endswith('.nii') or f.endswith('.nii.gz')]
    train_mask_files = [os.path.join(TRAIN_MASKS, f) for f in os.listdir(TRAIN_MASKS) if f.endswith('.nii') or f.endswith('.nii.gz')]
    X_train = load_data_2D(train_image_files, target_shape=TARGET_IMG_SHAPE)
    Y_train = load_data_2D(train_mask_files, target_shape=TARGET_IMG_SHAPE, categorical=True)

    print(f'Loaded {X_train.shape[0]} images of shape {X_train.shape[1:]}')

    print('Loading val data...')
    val_image_files = [os.path.join(VAL_IMAGES, f) for f in os.listdir(VAL_IMAGES) if f.endswith('.nii') or f.endswith('.nii.gz')]
    val_mask_files = [os.path.join(VAL_MASKS, f) for f in os.listdir(VAL_MASKS) if f.endswith('.nii') or f.endswith('.nii.gz')]
    X_val = load_data_2D(val_image_files, target_shape=TARGET_IMG_SHAPE)
    Y_val = load_data_2D(val_mask_files, target_shape=TARGET_IMG_SHAPE, categorical=True)

    print(f'Loaded {X_val.shape[0]} images of shape {Y_val.shape[1:]}')

if __name__ == '__main__':
    main()