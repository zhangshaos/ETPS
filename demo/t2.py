import sys
sys.path.append('../bin') # to found pyCCS
import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
from pyCCS import ccs, naive_segment, crf_segment, mrf_segment


img_dir     = '/mnt/f/XGrids/PHOTO/0_resized'
npy_dir   = '/mnt/f/XGrids/PHOTO/0_result/pred'
names       = [
    "085110107700026",
    "085110107700059",
    "085110107700068",
    "085110107700083",
    "085110107700146",
    "085110107700200",
    "085110107700240",
    "085110107700298",
    "085110107700345",
    "085110107700392",
    "085110107700551",
    "085110107700651",
    "085110107700695",
    "085110107700737",
    "085110107700807",
    "085110107700859",
    "085110107700865",
    "085110107701010",
    "085110107701050",
    "085110107701057",
    "085110107701119",
    "085110107701155",
    "085110107701208",
    "085110107701217",
    "085110107701225",
    "085110107701227",
    "085110107701272",
    "085110107701282",
    "085110107701308",
]


_COLOR_MAP_ = [
    np.array([255, 204, 153]),
    np.array([100, 100, 100]),
    np.array([0, 102, 0]),
    np.array([51, 255, 255]),
    np.array([153, 255, 51]),
    np.array([76, 153, 0]),
    np.array([255, 51, 51]),
    np.array([255, 0, 0]),
    np.array([0, 0, 204]),
    np.array([64, 64, 64]),
]


def draw_img(rgb_img: np.ndarray, mask: np.ndarray, num_class: int, alpha: float):
    assert 0 <= alpha and alpha <= 1
    if len(_COLOR_MAP_) < num_class:
        for t in range(num_class - len(_COLOR_MAP_)):
            _COLOR_MAP_.append(np.random.random(3) * 255)
    result = rgb_img.copy()
    for id in range(0, num_class):
        m = (mask == id)
        result[m] = rgb_img[m] * (1 - alpha) + _COLOR_MAP_[id] * alpha
    return result


def save_draw_img(name: str, rgb_img: np.ndarray, mask: np.ndarray, num_class: int, alpha: float):
    result = draw_img(rgb_img, mask, num_class, alpha)
    Image.fromarray(result).save(name)


if __name__ == '__main__':
    for i in range(len(names)):
        img_path = f'{img_dir}/{names[i]}.JPG'
        img = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        npy_path = f'{npy_dir}/{names[i]}.npy'
        sem = np.load(npy_path, allow_pickle=False)

        label       = ccs(img, verbose=False)
        sem_label   = naive_segment(label, img, sem)
        save_draw_img(f'{names[i]}_naive.png', img, sem_label, 10, 0.6)
        sem_label   = crf_segment(label, img, sem, verbose=False)
        save_draw_img(f'{names[i]}_crf.png', img, sem_label, 10, 0.6)
        sem_label   = mrf_segment(label, img, sem, verbose=False)
        save_draw_img(f'{names[i]}_mrf.png', img, sem_label, 10, 0.6)
    pass