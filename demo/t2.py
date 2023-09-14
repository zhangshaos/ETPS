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

if __name__ == '__main__':
    i = 10
    img_path = f'{img_dir}/{names[i]}.JPG'
    img = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.uint8)
    npy_path = f'{npy_dir}/{names[i]}.npy'
    sem = np.load(npy_path, allow_pickle=False)

    label       = ccs(img, verbose=False)
    sem_label   = naive_segment(label, img, sem)
    sem_label   = crf_segment(label, img, sem, verbose=True)
    sem_label   = mrf_segment(label, img, sem, verbose=True)
    pass