import os
import tensorflow as tf

from dataset import load_data_2D

BASE = os.path.dirname(__file__)
TRAINED_MODEL = os.path.join(BASE, "best_trained_unet.h5")
TEST_IMAGES = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_test')
TEST_MASKS = os.path.join(BASE, 'HipMRI_Study_open/keras_slices_data/keras_slices_seg_test')

TARGET_IMG_SHAPE = (128, 128)
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

    # Collect test image and mask file paths
    test_image_files = [os.path.join(TEST_IMAGES, f) for f in os.listdir(TEST_IMAGES) if f.endswith('.nii') or f.endswith('.nii.gz')]
    test_mask_files = [os.path.join(TEST_MASKS, f) for f in os.listdir(TEST_MASKS) if f.endswith('.nii') or f.endswith('.nii.gz')]
    X_test = load_data_2D(test_image_files, target_shape=TARGET_IMG_SHAPE)
    Y_test = load_data_2D(test_mask_files, target_shape=TARGET_IMG_SHAPE, categorical=True, num_classes=NUM_CLASSES)

    # Load model
    model = tf.keras.models.load_model(TRAINED_MODEL, compile=False)

    # Predict Y_test using X_test
    Y_pred = model.predict(X_test, batch_size=8)

    mean_dice = multiclass_dice_tf(Y_test, Y_pred, num_classes=6)

    print(f'Mean Dice on test set (all classes): {mean_dice:.4f}')

if (__name__ == "__main__"):
    main()
