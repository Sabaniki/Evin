# coding:utf-8

import cv2
import numpy as np
import image_funcs as imf
from k_means import run_kmeans

original_path = "Images/black_board.JPG"  # "Images/otaku_green.png"
new_path = "Images/black_board.png"


def print_image(str, image):
    print(str + " info: " + str(image.shape))  # 配列の次元を取得
    cv2.imshow(str, image)
    cv2.waitKey()


# 引数については->　http://opencv.jp/opencv-2.1/cpp/reading_and_writing_images_and_video.html
image = cv2.imread(original_path)
image = imf.scale(image, 500, 500)
image = imf.cv2pil(image)
image = list(image.convert('RGB').getdata())
image = imf.pil2cv(run_kmeans(image, 8))
print_image("raw_image_info", image)

green_min = np.array([110, 110, 100], np.uint8)  # ([0, 100, 0], np.uint8)  # 純色の緑に黒を追加したような暗い黒
green_max = np.array([140, 140, 150], np.uint8)  # ([30, 130, 30], np.uint8)  # R,B が少し加わり、緑の彩度が上がった明るい緑

threshold_otaku = cv2.bitwise_not(cv2.inRange(image, green_min, green_max))  # 二値化して黒白反転 二値化したので配列の次元が二次元になる
print_image("threshold_otaku", threshold_otaku)

mask = cv2.cvtColor(threshold_otaku, cv2.COLOR_GRAY2BGR)  # 二値化して二次元になった画像をBGRに変換してまた三次元にする
print_image("otalu_mask", mask)

new_image = cv2.addWeighted(image, 1, mask, 1, 0)

print_image("new_otaku", new_image)

cv2.imwrite(new_path, new_image)
