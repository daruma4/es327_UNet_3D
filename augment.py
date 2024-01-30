import albumentations as A
import cv2

def do_albumentations(transform, img_list, mask_list):
    """Does albumentation on all items in img_directory_list

    Args:
        img_directory_list (_type_): _description_
        mask_directory_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    idxs = range(len(img_list))
    augemented_raw_list = []
    augmented_mask_list = []
    for idx in idxs:
        curr_image = cv2.cvtColor(img_list[idx], cv2.COLOR_BGR2RGB)
        transformed = transform(image=curr_image, mask=mask_list[idx])
        transformed2 = transform(image=curr_image, mask=mask_list[idx])

        augemented_raw_list.append(curr_image)
        augemented_raw_list.append(transformed["image"])
        augemented_raw_list.append(transformed2["image"])
        augmented_mask_list.append(mask_list[idx])
        augmented_mask_list.append(transformed["mask"])
        augmented_mask_list.append(transformed2["mask"])

    return augemented_raw_list, augmented_mask_list