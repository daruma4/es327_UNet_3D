import albumentations as A
import cv2

def do_albumentations(transform, img_list, mask_list, slice_count):
    #say 384 images
    #24 3d slices
    #for each 3d slice
    #   load slice_count images
    #   transform
    #   add to aug_list
    slice_count_3d = range(len(img_list))
    augemented_raw_list = []
    augmented_mask_list = []
    for idx in slice_count_3d:
        slice_dict = {}
        for i in range(slice_count):
            if i is 0:
                slice_dict[f"image"] = cv2.cvtColor(img_list[idx][i], cv2.COLOR_BGR2RGB) ### first image has to be "image", no idx
                slice_dict[f"mask"] = cv2.cvtColor(mask_list[idx][i], cv2.COLOR_BGR2RGB) ### same with mask
                continue
            slice_dict[f"image{i}"] = cv2.cvtColor(img_list[idx][i], cv2.COLOR_BGR2RGB) ###
            slice_dict[f"mask{i}"] = cv2.cvtColor(mask_list[idx][i], cv2.COLOR_BGR2RGB) ###

        transformed = transform(**slice_dict)
        
        for i in transformed:
            if "image" in i:
                augemented_raw_list.append(transformed[i])
            elif "mask" in i:
                augmented_mask_list.append(transformed[i])
            else:
                raise Exception(f"ERROR: Could not identify key in transformed dict. {i} not in transformed.")

    return augemented_raw_list, augmented_mask_list