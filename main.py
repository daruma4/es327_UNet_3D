#Public
import json
import os
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models
import numpy as np

#Local
import niftiSave
import trainer
import predictor
import augmentator

#Current working directory of the project
ROOT_DIR = os.path.abspath("")
# Path to assets
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
# Path to store trained models
DEFAULT_LOGS_DIR = os.path.join(ASSETS_DIR, "trained_models")
# Path to images etc. for model
DATASET_DIR = os.path.join(ASSETS_DIR, "model_data")

################################
#||                          #||
#||       Nifti to PNG       #||
#||                          #||
################################
def main_nifti():
     PATH_RAW_IMAGE = os.path.join(DATASET_DIR, "raw\\image")
     PATH_RAW_MASK = os.path.join(DATASET_DIR, "raw\\mask")
     PATH_AUG_IMAGE = os.path.join(DATASET_DIR, "augmented\\image")
     PATH_AUG_MASK = os.path.join(DATASET_DIR, "augmented\\mask")
     PATH_NIFTI = os.path.join(DATASET_DIR, "nifti")
     PATH_NIFTI_META = os.path.join(PATH_NIFTI, "meta.json")

     #Load Nifti metadata and create niftisaver instance
     try:
          metadata = json.load(open(PATH_NIFTI_META))
     except FileNotFoundError:
          print(f"[ERROR] Metadata file not found at {PATH_NIFTI_META}")
     niftisaver = niftiSave.NiftiSave(path_save_image=PATH_RAW_IMAGE, path_save_mask=PATH_RAW_MASK, path_nifti=PATH_NIFTI, path_nifti_meta=PATH_NIFTI_META)
     nifti_folder_list = [os.path.join(PATH_NIFTI, each) for each in os.listdir(PATH_NIFTI) if each.endswith(".nii")]

     #Open each Nifti and save images + masks
     for nifti_path in nifti_folder_list:
          nib_nifti = nib.load(nifti_path)
          nifti_file_name = os.path.basename(nifti_path)

          raw_data = nib_nifti.get_fdata()[:,:,:,0,1] # all images
          data_iterable = niftisaver.image_array_to_iterable(raw_data) # make data iterable
          cut_ranges = metadata[nifti_file_name]["ranges"]
          data_iterable_ranged = niftisaver.range_crop(cut_ranges, data_iterable) # cuts to only range specified
          data_iterable_cropped = niftisaver.crop_images_iterable(data_iterable_ranged) # crop images

          mask_bool = metadata[nifti_file_name]["mask_bool"]
          save_path = niftisaver.path_save_image
          if mask_bool is True:
               save_path = niftisaver.path_save_mask

          
          niftisaver.save_images(save_path, metadata[nifti_file_name]["save_prefix"], data_iterable_cropped, mask_bool=mask_bool)

# ################################
# #||                          #||
# #||        Augmentor         #||
# #||                          #||
# ################################
# image_array=trainer.UNetTrainer.folder_to_array(PATH_TO_SAVE_RAW, 256, 256)
# mask_array=trainer.UNetTrainer.folder_to_array(PATH_TO_SAVE_MASK, 256, 256)

# aug_raw, aug_mask = augmentator.do_albumentations(img_list=image_array, mask_list=mask_array)
# niftiSave.NiftiSave.save_images(save_path=PATH_TO_SAVE_AUG_RAW, save_prefix="r", img_iterable=aug_raw)
# niftiSave.NiftiSave.save_images(save_path=PATH_TO_SAVE_AUG_MASK, save_prefix="m", img_iterable=aug_mask)
# ################################
# #||                          #||
# #||         Trainer          #||
# #||                          #||
# ################################
# unetObj = trainer.UNetTrainer(img_height=256, img_width=256, img_channels=1,epochs=10)
# raw_images = unetObj.folder_to_array(PATH_TO_SAVE_AUG_RAW, unetObj.img_width, unetObj.img_height)
# mask_images = unetObj.folder_to_array(PATH_TO_SAVE_AUG_MASK, unetObj.img_width, unetObj.img_height)
# print("[L] Images to Array completed.")

# myModel = unetObj.create_unet_model()
# print("[L] Model created.")
# # myModel.compile(optimizer="adam", loss=unetObj.dice_coef_loss, metrics=[unetObj.iou, unetObj.dice_coef, unetObj.precision, unetObj.recall, unetObj.accuracy])
# myModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) 
# print("[L] Model compile created.")

# print("[L] Training model...")
# # want loss to be 0
# myModel_trained = myModel.fit(x=raw_images, y=mask_images, validation_split=0.25, batch_size= 16, epochs=unetObj.epochs, shuffle=True)
# print("[L] Model trained.")
# myModel.save("2Dunet.h5")
# print("[L] Model saved.")
# ################################
# #||                          #||
# #||        Predictor         #||
# #||                          #||
# ################################
# predictorObj = predictor.predictor(model=models.load_model(MODEL_NAME, compile=False), 
#                     image_array=trainer.UNetTrainer.folder_to_array(PATH_TO_SAVE_RAW, 256, 256), 
#                     mask_array=trainer.UNetTrainer.folder_to_array(PATH_TO_SAVE_MASK, 256, 256))

# ran_image, ran_mask, predicted_mask = predictorObj.random_predict()
# fig, subplots = plt.subplots(3, 1)
# subplots[0].imshow(ran_image, cmap='gray')
# subplots[0].set_title(f"Image")
# subplots[1].imshow(ran_mask, cmap='gray')
# subplots[1].set_title(f"Mask")
# subplots[2].imshow(np.reshape(predicted_mask, (256,256,1)), cmap='gray')
# subplots[2].set_title(f"Predicted Mask")
# for subplot in subplots:
#      subplot.set_xticks([])
#      subplot.set_yticks([])
# plt.show()