import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dataset import load_data_2D

BASE = os.path.dirname(__file__)
TRAINED_MODEL = os.path.join(BASE, "best_trained_unet.h5")
TEST_IMAGES = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_test')
TEST_MASKS = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_seg_test')

TARGET_IMG_SHAPE = (256, 128)
NUM_CLASSES = 6

def multiclass_dice_tf(y_true, y_pred, num_classes=6, epsilon=1e-6):
    """
    Computes mean Dice coefficient for multi-class segmentation using hard class predictions.

    y_true, y_pred: (batch, H, W, num_classes) one-hot arrays
    y_pred: can be softmax probabilities or one-hot
    """
    # Convert softmax probabilities to hard predictions (one-hot encoding)
    if y_pred.shape[-1] == num_classes:
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), num_classes)
    # Ensure ground truth and predictions are float tensors
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    dice_scores = []
    # Loop over each class to compute Dice score
    for c in range(num_classes):
        # Flatten the mask for class c
        y_true_c = tf.reshape(y_true[..., c], [-1])
        y_pred_c = tf.reshape(y_pred[..., c], [-1])
        # Compute intersection between prediction and ground truth for class c
        intersection = tf.reduce_sum(y_true_c * y_pred_c)
        # Compute Dice score for class c (add epsilon for numerical stability)
        dice = (2. * intersection + epsilon) / (tf.reduce_sum(y_true_c) + tf.reduce_sum(y_pred_c) + epsilon)
        # Store Dice score for this class
        dice_scores.append(dice)
    # Return the mean Dice score across all classes
    return tf.reduce_mean(dice_scores)

def main():

    if not os.path.exists(TRAINED_MODEL):
        raise FileNotFoundError(f"Model file not found at: {TRAINED_MODEL}")

    # Collect test image and mask file paths & load using dataset.py
    test_image_files = [os.path.join(TEST_IMAGES, f) for f in os.listdir(TEST_IMAGES) if f.endswith('.nii') or f.endswith('.nii.gz')]
    test_mask_files = [os.path.join(TEST_MASKS, f) for f in os.listdir(TEST_MASKS) if f.endswith('.nii') or f.endswith('.nii.gz')]
    X_test = load_data_2D(test_image_files, target_shape=TARGET_IMG_SHAPE)
    Y_test = load_data_2D(test_mask_files, target_shape=TARGET_IMG_SHAPE, categorical=True, num_classes=NUM_CLASSES)

    # Load model
    model = tf.keras.models.load_model(TRAINED_MODEL, compile=False)

    # Predict Y_test using X_test
    Y_pred = model.predict(X_test, batch_size=8)

    # Get mean Dice score of model predictions
    mean_dice = multiclass_dice_tf(Y_test, Y_pred, num_classes=6)

    print(f'Mean Dice on test set (all classes): {mean_dice:.4f}')

    # Plot 5 test images against ground truth and predicted masks
    num_to_show = min(5, len(X_test))
    for i in range(num_to_show):
        # Get image at number i
        img = np.squeeze(X_test[i])

        # Convert the masks to one-hot encoded 
        true_mask = np.argmax(np.squeeze(Y_test[i]), axis=-1)
        pred_mask = np.argmax(np.squeeze(Y_pred[i]), axis=-1)

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1) # Plot No1
        plt.imshow(img, cmap='gray') # Show MRI Slice and Colour
        plt.title('MRI Slice')
        plt.axis('off')

        plt.subplot(1, 3, 2) # Plot No2
        plt.imshow(img, cmap='gray')
        plt.imshow(true_mask, alpha=0.5, cmap='tab10', vmin=0, vmax=5) # Overlay true mask on slice with transparency
        plt.title('Ground Truth Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3) # Plot No3
        plt.imshow(img, cmap='gray')
        plt.imshow(pred_mask, alpha=0.5, cmap='tab10', vmin=0, vmax=5)
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.suptitle(f'Test Masks {i+1} | Classes: {", ".join(class_names)}') # Subtitle
        plt.tight_layout()
        plt.show()

if (__name__ == "__main__"):
    main()
