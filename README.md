# Handwriting-Removal-DIS
My effort into improving handwriting removal throught the new DIS (Dichotomous Image Segmentation)

## Related Research
AndSonder has also done research and experimentaion on the same subject but using deeplabv3+ to segment the handwriting.

This is a link to his repo: [https://github.com/AndSonder/HandWritingEraser-Pytorch](https://github.com/AndSonder/HandWritingEraser-Pytorch)

HUGE THANKS to them for providing the segmentation datasets labeled with background blue, printed characters green, and handwriting in red.

## Dataset
The original dataset is in Baidu Web Storage and is a segmentation dataset, unlike a background removal dataset.

Therefore, after some processing, I generated a background-removal dataset. It is available in Huggingface: [https://huggingface.co/datasets/Inoob/HandwritingSegmentationDataset](https://huggingface.co/datasets/Inoob/HandwritingSegmentationDataset).

The relavent contents of the repo is listed:

```
|- train.zip
|- val.zip
```

After unzipping train.zip and val.zip, the file tree should look like:

```
|-train
|    |-gt
|    |  |- dehw_train_00714.png
|    |  |- dehw_train_00715.png
|    |  ...
|    |-im
|    |  |- dehw_train_00714.jpg
|    |  |- dehw_train_00715.jpg
|-val
|    |-gt
|    |  |- dehw_train_00000.png
|    |  |- dehw_train_00001.png
|    |  ...
|    |-im
|    |  |- dehw_train_00000.png
|    |  |- dehw_train_00001.png
```

the ```gt``` folder is masks. With the background masked in black, and the handwriting masked as white (a.k.a ground truth data).

the ```im``` folder is the normal image of the handwriting dataset.

The code that was used to generate the dataset in the Huggingface Repo is ```create_masks.py```

## Training

I used the ```train_valid_inference_main.py``` from [DIS](https://github.com/xuebinqin/DIS) with my own dataset and training batch size.

You can scale the batch size up if you have enough memory.
