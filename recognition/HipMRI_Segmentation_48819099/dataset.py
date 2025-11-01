import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
from skimage.transform import resize



def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert a 2D array of integer labels into one-hot channels.

    Example: arr values in {0,1,2} -> output shape (H,W,3)
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1

    return res


def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False, target_shape=(256, 256)):
    """
    Load 2D medical image data from a list of NIfTI filenames. 
        This function pre - allocates 4D arrays for conv2d to avoid excessive memory & usage.
        This code that has been previously provided has been updated to allow for image remapping,
            this allows for faster testing of the model, as well as scaling images to consistent dimensions.
        THIS CODE DOES NOT ACCOUNT FOR 0 IMAGES CASE.
        

    Parameters:
      imageNames: list of file paths (NIfTI expected)
      normImage: normalize each image (zero-mean, unit-std)
      categorical: if True produce one-hot channels for labels
      dtype: numpy dtype for output
      getAffines: if True also return the list of affines
      early_stop: if True stop after ~20 images (quick test)
      target_shape: (rows, cols) to resize to

    Returns:
      images (and optionally affines)
    """
    affines = []


    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(resize(first_case, target_shape, preserve_range=True), dtype=dtype)
        rows, cols, channels = first_case.shape
        print(channels)
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        first_case = resize(first_case, target_shape, preserve_range=True)
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data

        inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np. linalg . norm ( inImage )
            #~ inImage = 255. * inImage / inImage . max ()
            inImage = (inImage - inImage.mean()) / inImage.std()

        inImage = resize(inImage, target_shape, preserve_range=True)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


if __name__ == '__main__':
    """
    Runs a small test of the code.

    Loads the dataset using the load_data_2D method, then prints the numbers to check if reasonable.
    """

    # load normal image data
    base = os.path.dirname(__file__)
    train_folder = os.path.join(base, 'HipMRI_Study_open\keras_slices_data\keras_slices_train')
    image_names = [os.path.join(train_folder, fname) for fname in os.listdir(train_folder) if fname.endswith('.nii') or fname.endswith('.nii.gz')]
    images = load_data_2D(image_names, categorical=False, target_shape=(256, 256))

    # load mask data test
    base = os.path.dirname(__file__)
    train_folder = os.path.join(base, 'HipMRI_Study_open\keras_slices_data\keras_slices_seg_train')
    image_names = [os.path.join(train_folder, fname) for fname in os.listdir(train_folder) if fname.endswith('.nii') or fname.endswith('.nii.gz')]
    images = load_data_2D(image_names, categorical=True, target_shape=(256, 256))


    # output tests of loaded data (counts / possibly images)
    print(f'Loaded {images.shape[0]} images of shape {images.shape[1:]}')