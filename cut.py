import os

import cv2
import numpy as np
from tqdm import tqdm
# 调节亮度测试    目前调节 -60 效果是最好的情况
import os

import cv2
import numpy as np
from tqdm import tqdm


def compute_weight(img):
    b1, g1, r1 = cv2.split(img)
    b1_mean, g1_mean, r1_mean = np.mean(b1), np.mean(g1), np.mean(r1)
    mean = 0.299 * r1_mean + 0.587 * g1_mean + 0.114 * b1_mean
    # print(mean)
    return mean


def adjust(file_ame, save_path):
    img = cv2.imread(file_ame)
    img2 = np.zeros(img.shape, img.dtype)
    mean = compute_weight(img)
    while mean > 70:
        if mean > 120:
            img = cv2.addWeighted(img, 1, img2, 2, -60)
        elif mean > 100:
            img = cv2.addWeighted(img, 1, img2, 2, -40)
        elif mean > 80:
            img = cv2.addWeighted(img, 1, img2, 2, -20)
        else:
            img = cv2.addWeighted(img, 1, img2, 2, -10)
        mean = compute_weight(img)
    # cv2.imwrite(save_path, img)
    return img


def test(threshold, img, save_name):
    # img = cv2.imread(file_name)

    lower = np.array([0, 0, 0])
    upper = np.array([100, 255, 120])

    gray_img = cv2.inRange(img, lower, upper)

    cv2.imwrite(save_name, gray_img)
    contours, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= threshold:  # 默认值： 80000
            cv_contours.append(contour)
        else:
            continue

    cv2.fillPoly(gray_img, cv_contours, (255, 255, 255))
    result = cv2.bitwise_and(img, img, mask=gray_img)

    cv2.imwrite(save_name, result)


file_root = r'/root/autodl-tmp/fasterrcnn/tools/cyqp'
save_root = r'/root/autodl-tmp/fasterrcnn/tools/cyqp_test/test01'
file_name_list = os.listdir(file_root)
# print(file_name_list)
thresholds = [80000, 40000, 20000, 10000, 8000, 4000, 2000, 1000, 800, 600, 400]
for file_name in tqdm(file_name_list):
    file_path = os.path.join(file_root, file_name)
    file_save_name = os.path.join(save_root, file_name)
    img = adjust(file_path, file_save_name)
    for threshold in thresholds:
        print(rf'file:{file_name}    -','threshold:',threshold)
        name, types = file_save_name.rsplit('.', 1)
        test(threshold, img, rf'{name}_{threshold}.{types}')
