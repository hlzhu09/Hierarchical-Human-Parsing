import numpy as np
from PIL import Image

# colour map
label_colours = [(0, 0, 0)
                 # 0=Background
    , (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51), (255, 85, 0)
                 # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
    , (0, 0, 85), (0, 119, 221), (85, 85, 0), (0, 85, 85), (85, 51, 0)
                 # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
    , (52, 86, 128), (0, 128, 0), (0, 0, 255), (51, 170, 221), (0, 255, 255)
                 # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
    , (85, 255, 170), (170, 255, 85), (255, 255, 0), (255, 170, 0)]
# 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe


pascal_person = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128)]


def decode_predictions(preds, num_images=4, num_classes=20):
    """Decode batch of segmentation masks.
    """
    preds = preds.data.cpu().numpy()
    n, h, w = preds.shape
    assert n >= num_images
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(preds[i, 0]), len(preds[i])))
        pixels = img.load()
        for j_, j in enumerate(preds[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


def inv_preprocess(imgs, num_images=4):
    """Inverse preprocessing of the batch of images.
    """
    mean = (104.00698793, 116.66876762, 122.67891434)
    imgs = imgs.data.cpu().numpy()
    n, c, h, w = imgs.shape
    assert n >= num_images
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (np.transpose(imgs[i], (1, 2, 0)) + mean)[:, :, ::-1].astype(np.uint8)
    return outputs
