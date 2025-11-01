import os
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import load_data_2D
from modules import unet_2d

BASE = os.path.dirname(__file__)
TRAIN_IMAGES = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_train')
TRAIN_MASKS = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_seg_train')
VAL_IMAGES = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_validate')
VAL_MASKS = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate')

NUM_CLASSES = 6

MODEL_SAVE_OUT = os.path.join(BASE, 'best_trained_unet.h5')

EPOCHS = 10
BATCH_SIZE = 64

TARGET_IMG_SHAPE = (256, 128)

def main():
    # base file path
    base = os.path.dirname(__file__)

    # 
    print('Loading train data...')
    train_image_files = [os.path.join(TRAIN_IMAGES, f) for f in os.listdir(TRAIN_IMAGES) if f.endswith('.nii') or f.endswith('.nii.gz')]
    train_mask_files = [os.path.join(TRAIN_MASKS, f) for f in os.listdir(TRAIN_MASKS) if f.endswith('.nii') or f.endswith('.nii.gz')]
    X_train = load_data_2D(train_image_files, target_shape=TARGET_IMG_SHAPE)
    Y_train = load_data_2D(train_mask_files, target_shape=TARGET_IMG_SHAPE, categorical=True, num_classes=NUM_CLASSES)

    print(f'Loaded {X_train.shape[0]} images of shape {X_train.shape[1:]}')

    print('Loading val data...')
    val_image_files = [os.path.join(VAL_IMAGES, f) for f in os.listdir(VAL_IMAGES) if f.endswith('.nii') or f.endswith('.nii.gz')]
    val_mask_files = [os.path.join(VAL_MASKS, f) for f in os.listdir(VAL_MASKS) if f.endswith('.nii') or f.endswith('.nii.gz')]
    X_val = load_data_2D(val_image_files, target_shape=TARGET_IMG_SHAPE)
    Y_val = load_data_2D(val_mask_files, target_shape=TARGET_IMG_SHAPE, categorical=True, num_classes=NUM_CLASSES)

    print(f'Loaded {X_val.shape[0]} images of shape {Y_val.shape[1:]}')

    model = unet_2d(input_size=(TARGET_IMG_SHAPE[0], TARGET_IMG_SHAPE[1], 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_OUT, monitor='val_loss', mode='min', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=12, verbose=1)
    ]

    model_fit_res = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True
    )

    # Plot training and validation loss
    history = model_fit_res.history
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()