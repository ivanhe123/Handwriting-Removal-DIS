import os
import cv2
import numpy as np
from tqdm import tqdm

def process_image(input_path, output_path):

    img = cv2.imread(input_path)

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
