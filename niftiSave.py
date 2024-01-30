#Opening Nifti file formats
import nibabel as nib

#plotting and converting to array modules
import numpy as np

#image saving
import cv2
import os

def crop_center(img, cropx, cropy):
    """Centre crops image

    Args:
        img (numpy.ndarray): Image 2D array
        cropx (int): Max width x
        cropy (int): Max height x

    Returns:
        numpy.ndarray: Cropped image 2D array
    """
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def image_array_to_iterable(img_array):
    """Converts an array containing many 2D image data into an iterable version i.e. a list [n,1] containg each image [img_height, img_width]

    Args:
        img_array (numpy.ndarray): 2D image array
    """
    returnList = []
    num = img_array.shape[2]
    for idx in range(num):
        returnList.append(img_array[:, :, idx])

    return returnList

def crop_images_iterable(img_iterable):
    """Center crops all images in iterable

    Args:
        img_iterable (list): List containing 2D images

    Returns:
        list: cropped image iterable
    """
    returnList = []
    for image in img_iterable:
        returnList.append(crop_center(np.fliplr(np.rot90(image)), 256, 256)) # for raw

    return returnList

def range_crop(ranges, img_iterable):
    """Returns the img_iterable only in the ranges specified

    Args:
        ranges (List[int]): List of ranges in form [start, end]
        img_iterable (_type_): _description_

    Returns:
        _type_: _description_
    """
    workingList = []
    for range in ranges:
        minR = range[0]
        maxR = range[1]
        for img in img_iterable[minR:maxR]:
            workingList.append(img)
    return workingList
    
    
def save_images(save_path, save_prefix, img_iterable, mask_bool):
    """Saves each element of img_iterable to save_path location with the prefix save_prefix

    Args:
        save_path (str): _description_
        save_prefix (str): _description_
        img_iterable (np.ndarray): _description_
        mask_bool (bool): _description_
    """
    for idx, img in enumerate(img_iterable):
        save_file_path = os.path.join(save_path, f"{save_prefix}_{idx}.png")
        if mask_bool is True:
            img = cv2.convertScaleAbs(img, alpha=(255.0)) # correct range of image for saving mask
        cv2.imwrite(save_file_path, img)


def load_images(folder, img_width=256, img_height=256, normalize=False):
    """Returns images in a folder as a Numpy array

    Args:
        folder (string): folder path that contains images

    Returns:
        image_list (np.ndarray): array of images
    """
    image_list = []
    image_folder_list = [os.path.join(folder, each) for each in os.listdir(folder)]
    for img_path in image_folder_list:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if normalize is True:
                img = img / 255 # Normalise images for training model
            image_list.append(img)
    image_list = np.array(image_list)
    image_list = np.reshape(image_list, (-1, img_width, img_height, 1)).astype(np.float32)

    return image_list