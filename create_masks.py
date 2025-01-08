import os
import cv2
import numpy as np
from tqdm import tqdm
def resize_image(image, output_path, size=1024):
    
    height, width = image.shape[:2]

    if height > size or width > size:
        # Calculate the scale factor
        if height > width:
            scale_factor = size / height
        else:
            scale_factor = size / width

        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

        return resized_image

    else:
        return image

def process_image(input_path, output_path):

    img = cv2.imread(input_path)

    img = resize_image(img)

    mask = cv2.inRange(img, (0, 0, 255), (0, 0, 255))

    img[mask != 0] = [255, 255, 255]

    img[mask == 0] = [0, 0, 0]

    cv2.imwrite(output_path, img)


mask_dir = "mask"
output_dir = "training_sets/gt"
output_val_dir = "val_set/gt"

validation = 100
cnt = 0
for file in tqdm(os.listdir(mask_dir)):
    cnt += 1
    if cnt > 814 - validation:
        process_image(mask_dir + "/" + file, output_val_dir + "/" + file)
    else:
        process_image(mask_dir+"/"+file, output_dir+"/"+file)
