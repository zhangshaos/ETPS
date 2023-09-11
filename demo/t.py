import sys 
sys.path.append('../bin') # to found pyCCS
import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
from pyCCS import ccs


# img_path = "../demo/test_img/texture_compo.png"
# img_path = "../demo/test_img/UI_seq17_000700.jpg"
img_path = "../demo/test_img/6h00002.jpg"
# img_path = "../demo/test_img/UD_000261.jpg"

if __name__ == '__main__':
    img = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.uint8)
    label = ccs(img)
    result = np.asarray(mark_boundaries(img, label) * 255, dtype=np.uint8)
    Image.fromarray(result).save("t.jpg")
    pass