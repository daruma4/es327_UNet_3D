#Opening Nifti file formats
import nibabel as nib

#plotting and converting to array modules
import numpy as np

#image saving
import cv2
import os

class NiftiSave:
    def __init__(self, path_save_image, path_save_mask, path_nifti, path_nifti_meta):
        self.path_save_image = path_save_image
        self.path_save_mask = path_save_mask
        self.path_nifti = path_nifti
        self.path_nifti_meta = path_nifti_meta

    def crop_center(self, img, cropx, cropy):
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

    def image_array_to_iterable(self, img_array):
        """Converts an array containing many 2D image data into an iterable version i.e. a list [n,1] containg each image [img_height, img_width]

        Args:
            img_array (numpy.ndarray): 2D image array
        """
        returnList = []
        num = img_array.shape[2]
        for idx in range(num):
            returnList.append(img_array[:, :, idx])

        return returnList

    def crop_images_iterable(self, img_iterable):
        """Center crops all images in iterable

        Args:
            img_iterable (list): List containing 2D images

        Returns:
            list: cropped image iterable
        """
        returnList = []
        for image in img_iterable:
            returnList.append(self.crop_center(np.fliplr(np.rot90(image)), 256, 256)) # for raw

        return returnList

    @staticmethod
    def save_images(save_path, save_prefix, img_iterable, mask_bool):
        """Saves images to save_path with save_prefix

        Args:
            save_path (_type_): _description_
            save_prefix (_type_): _description_
            img_iterable (_type_): _description_
        """
        for idx, img in enumerate(img_iterable):
            save_file_path = os.path.join(save_path, f"{save_prefix}_{idx}.png")
            if mask_bool is True:
                img = cv2.convertScaleAbs(img, alpha=(255.0)) # correct range of image for saving mask
            cv2.imwrite(save_file_path, img)

    def range_crop(self, ranges, img_iterable):
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