import cv2
from PIL import Image
import numpy as np


def scale(img, width, height, keep_aspect=True):
    if keep_aspect:
        my_scale = max(width / img.shape[1], height / img.shape[0])
        return cv2.resize(img, dsize=None, fx=my_scale, fy=my_scale)
    else:
        return cv2.resize(img, (width, height))


def cv2pil(image_cv):
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cv)
    image_pil = image_pil.convert('RGB')

    return image_pil


def pil2cv(image):
    """ PIL型 -> OpenCV型 """
    new_image = np.array(image)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return new_image
