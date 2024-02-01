#Public
import json
import os
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models
import numpy as np
import albumentations

#Local
import niftiSave
import unet
import predictor
import augment

#Current working directory of the project
ROOT_DIR = os.path.abspath("")
# Path to assets
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
# Path to store trained models
DEFAULT_LOGS_DIR = os.path.join(ASSETS_DIR, "trained_models")
# Path to images etc. for model
DATASET_DIR = os.path.join(ASSETS_DIR, "model_data")

PATH_RAW_IMAGE_2D = os.path.join(DATASET_DIR, "raw_2d\\image")
PATH_RAW_MASK_2D = os.path.join(DATASET_DIR, "raw_2d\\mask")
PATH_RAW_IMAGE_3D = os.path.join(DATASET_DIR, "raw_3d\\image")
PATH_RAW_MASK_3D = os.path.join(DATASET_DIR, "raw_3d\\mask")
PATH_AUG_IMAGE = os.path.join(DATASET_DIR, "augmented_3d\\image")
PATH_AUG_MASK = os.path.join(DATASET_DIR, "augmented_3d\\mask")
PATH_NIFTI = os.path.join(DATASET_DIR, "nifti")
PATH_NIFTI_META = os.path.join(PATH_NIFTI, "meta.json")
################################
#||                          #||
#||       Nifti to PNG       #||
#||                          #||
################################
def main_nifti():
     #Load Nifti metadata and create niftisaver instance
     try:
          metadata = json.load(open(PATH_NIFTI_META))
     except FileNotFoundError:
          print(f"[ERROR] Metadata file not found at {PATH_NIFTI_META}")
     nifti_folder_list = [os.path.join(PATH_NIFTI, each) for each in os.listdir(PATH_NIFTI) if each.endswith(".nii")]

     #Open each Nifti and save images + masks
     for nifti_path in nifti_folder_list:
          nib_nifti = nib.load(nifti_path)
          nifti_file_name = os.path.basename(nifti_path)

          raw_data = nib_nifti.get_fdata()[:,:,:,0,1] # all images
          data_iterable = niftiSave.image_array_to_iterable(raw_data) # make data iterable
          cut_ranges = metadata[nifti_file_name]["ranges"]
          data_iterable_ranged = niftiSave.range_crop(cut_ranges, data_iterable) # cuts to only range specified
          data_iterable_cropped = niftiSave.crop_images_iterable(data_iterable_ranged) # crop images

          mask_bool = metadata[nifti_file_name]["mask_bool"]
          save_path = PATH_RAW_IMAGE_2D
          if mask_bool is True:
               save_path = PATH_RAW_MASK_2D

          
          niftiSave.save_images(save_path, metadata[nifti_file_name]["save_prefix"], data_iterable_cropped, mask_bool=mask_bool)

################################
#||                          #||
#||        3D Parser         #||
#||                          #||
################################
def main_3dparser(slice_count=16):
     raw2d_dict = {}
     raw2d_folder_list = [os.path.join(PATH_RAW_IMAGE_2D, each) for each in sorted(os.listdir(PATH_RAW_IMAGE_2D))]
     for path in raw2d_folder_list:
          prefix = os.path.basename(path).split("_")[0]
          if prefix not in raw2d_dict:
               raw2d_dict[prefix] = {}
               raw2d_dict[prefix]["files"] = []
          raw2d_dict[prefix]["files"].append(path)
     for prefix in raw2d_dict:
          raw2d_dict[prefix]["files"] = sorted(
                                                  raw2d_dict[prefix]["files"], 
                                                  key = lambda i: int(os.path.basename(i).split("_")[1].split(".")[0])
                                               ) # Natural Sort the files
          raw2d_dict[prefix]["file_count"] = len(raw2d_dict[prefix]["files"])
          raw2d_dict[prefix]["file_idx_floor"] = (raw2d_dict[prefix]["file_count"] // slice_count) * slice_count
          raw2d_dict[prefix]["files"] = raw2d_dict[prefix]["files"][:raw2d_dict[prefix]["file_idx_floor"]] # Only keep paths that allow for exact slice_count slices of a scan

          #Open all files from list
          for image_path in raw2d_dict[prefix]["files"]:
               img = niftiSave.load_path_as_img(image_path)
               niftiSave.save_img_to_path(img=img, path=os.path.join(PATH_RAW_IMAGE_3D, os.path.basename(image_path)), mask_bool=False)
               #Save to 3D image dir
               #Open all masks
               mask_file_name = f"m{os.path.basename(image_path)[1:]}"
               mask_2d_path = os.path.join(PATH_RAW_MASK_2D, mask_file_name)
               mask = niftiSave.load_path_as_img(mask_2d_path)
               #Save to 3D mask dir
               niftiSave.save_img_to_path(img=mask, path=os.path.join(PATH_RAW_MASK_3D, mask_file_name), mask_bool=True)

################################
#||                          #||
#||        Augmentor         #||
#||                          #||
################################
def main_augmentation(slice_count=16):
     # edit so augments slice_count at a time (e.g. 16 images augmented the same)
     additional_targets = {}
     for i in range(1, slice_count):
          additional_targets[f"image{i}"] = "image"
          additional_targets[f"mask{i}"] = "mask"
     AUGMENTATIONS_LIST = albumentations.Compose(
          [
               albumentations.Blur(blur_limit=15, p=0.5),
               albumentations.HorizontalFlip(p=0.5),
               albumentations.VerticalFlip(p=0.5),
               albumentations.RandomRotate90(p=0.5),
          ],
          additional_targets=additional_targets
     )

     image_array=niftiSave.load_folder_3d(PATH_RAW_IMAGE_3D)
     mask_array=niftiSave.load_folder_3d(PATH_RAW_MASK_3D)


     aug_raw, aug_mask = augment.do_albumentations(transform=AUGMENTATIONS_LIST, img_list=image_array, mask_list=mask_array, slice_count=slice_count)
     niftiSave.save_images(save_path=PATH_AUG_IMAGE, save_prefix="r", img_iterable=aug_raw, mask_bool=False)
     niftiSave.save_images(save_path=PATH_AUG_MASK, save_prefix="m", img_iterable=aug_mask, mask_bool=True)

################################
#||                          #||
#||         Trainer          #||
#||                          #||
################################
def main_trainer(img_height=256, img_width=256, img_channels=1, epochs=100, filter_num=32, batch_size=1, learning_rate=0.0001):
     #Should setup to change filter_num, batch_size and learning_rate
     unetObj = unet.unet_model(filter_num=filter_num, img_height=img_height, img_width=img_width, img_channels=img_channels, epochs=epochs)
     raw_images = niftiSave.load_folder_3d(PATH_AUG_IMAGE, normalize=True)
     raw_masks = niftiSave.load_folder_3d(PATH_AUG_MASK, normalize=True)

     #Prepare model
     myModel = unetObj.create_unet_model(filter_num=filter_num)
     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
     loss = unetObj.j_dice_coef_loss
     ## Investigation needed - why does it train on J_dice_coef_loss and not g...

     ## SHOULD EXPERIMENT WITH SMOOTHING VALUE IN J_IOU_LOSS 
     metrics = [unetObj.g_dice_coef_loss, unetObj.j_dice_coef_loss, unetObj.g_iou, unetObj.j_iou, unetObj.g_iou_loss]
     myModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

     #Prepare callbacks
     earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
     reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3,verbose=1)


     #Do fit
     myModel_trained = myModel.fit(x=raw_images, y=raw_masks, validation_split=0.25, batch_size=batch_size, epochs=unetObj.epochs, shuffle=False, validation_batch_size=1, callbacks=[earlystopper, reduce_lr])
     myModelSavePath = os.path.join(DEFAULT_LOGS_DIR, f"3d_fn{filter_num}-bs{batch_size}-lr{learning_rate}.h5")
     myModelHistorySavePath = os.path.join(DEFAULT_LOGS_DIR, f"3d_fn{filter_num}-bs{batch_size}-lr{learning_rate}.npy")
     myModel.save(myModelSavePath)
     np.save(myModelHistorySavePath, myModel_trained.history)

# ################################
# #||                          #||
# #||        Predictor         #||
# #||                          #||
# ################################
def predict(model_path: str):
     predictorObj = predictor.predictor(model=models.load_model(model_path, compile=False), 
                         image_array=niftiSave.load_images(PATH_AUG_IMAGE, normalize=True), 
                         mask_array=niftiSave.load_images(PATH_AUG_MASK, normalize=True))

     ran_image, ran_mask, predicted_mask = predictorObj.random_predict()
     fig, subplots = plt.subplots(3, 1)
     subplots[0].imshow(ran_image, cmap='gray')
     subplots[0].set_title(f"Image")
     subplots[1].imshow(ran_mask, cmap='gray')
     subplots[1].set_title(f"Mask")
     subplots[2].imshow(np.reshape(predicted_mask, (256,256,1)), cmap='gray')
     subplots[2].set_title(f"Predicted Mask")
     for subplot in subplots:
          subplot.set_xticks([])
          subplot.set_yticks([])
     plt.show()

# predict(os.path.join(DEFAULT_LOGS_DIR, "fn32-bs16-lr0.0001.h5"))