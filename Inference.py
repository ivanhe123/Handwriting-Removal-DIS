import os
import time
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from skimage import io
import time
from glob import glob
from tqdm import tqdm
import cv2
import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize


def resize_image(image, size=1024):
    height, width = image.shape[:2]

    if height > size or width > size:

        if height > width:
            scale_factor = size / height
        else:
            scale_factor = size / width

        # Resize the image
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

        # Save the resized image

        print(f"Image resized to {new_dimensions}")
        return resized_image
    else:
        print("Image is already within the desired size.")
        return image
def runer(f):
    im = cv2.imread(os.path.join(dataset_path, f))
    im = resize_image(im)
    temp = np.ones((1024, 1024, 3))

    if len(im.shape) < 3:
        im = np.stack([im] * 3, axis=-1)  # Convert grayscale to RGB
    h, w = im.shape[0], im.shape[1]
    temp[:h, :w] = im
    im = temp
    im_shp = im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    if torch.cuda.is_available():
        image = image.cuda()
    result = net(image)
    result = torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    result = result.unsqueeze(0) if result.dim() == 2 else result  # Ensure result has 3 channels
    result = result.repeat(3, 1, 1) if result.shape[0] == 1 else result
    result = 1 - result  # Invert the mask here


    if torch.cuda.is_available():
        result = result.cuda()  # Move result to GPU if available

    #im_name = im_path.split('\\')[-1].split('.')[0]

    # Resize the image to match result dimensions
    image_resized = F.upsample(image, size=result.shape[1:], mode='bilinear')

    # Ensure both tensors are 3D
    image_resized = image_resized.squeeze(0) if image_resized.dim() == 4 else image_resized
    result = result.squeeze(0) if result.dim() == 4 else result

    # Apply threshold to result to ensure only pure black or white pixels
    threshold = 0.65 # Adjust as needed
    result[result < threshold] = 0
    result[result >= threshold] = 1

    distance = np.sqrt(np.sum((im - [255, 255, 255]) ** 2, axis=-1))

    threshold1 = 200
    # Create a mask where the distance is less than the threshold
    mask = distance < threshold1

    # Convert mask to uint8
    mask = mask.astype(np.uint8) * 255

    mask = np.stack([mask] * 3, axis=-1)

    result = (result.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
    # result=result.cpu().numpy().astype(np.uint8)
    # io.imsave(result_path + im_name + "_foreground.png", foreground)
    wite = np.ones_like(im) * 255
    cropped = np.where(result == 0, wite, mask)
    #show_pic(cropped)
    cv2.imwrite(result_path + f, cropped[:h, :w])
    return cropped[:h, :w]

if __name__ == "__main__":
    dataset_path="./val/im"  #Your dataset path
    model_path="./isnet.pth"  # the model path
    result_path="./res/"  #The folder path that you want to save the results

    net = ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cuda"))
    net.eval()

    im_list = glob(dataset_path + "/*.jpg") + glob(dataset_path + "/*.JPG") + glob(dataset_path + "/*.jpeg") + glob(
        dataset_path + "/*.JPEG") + glob(dataset_path + "/*.png") + glob(dataset_path + "/*.PNG") + glob(
        dataset_path + "/*.bmp") + glob(dataset_path + "/*.BMP") + glob(dataset_path + "/*.tiff") + glob(
        dataset_path + "/*.TIFF")


    with torch.no_grad():
        for f in tqdm(os.listdir(dataset_path)):
            runer(f)
