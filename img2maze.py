import cv2
from PIL import Image
import numpy as np
import sys
import os

if __name__ == "__main__":
    print("Starting image to maze conversion...")
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        sys.exit("Please imput a valid path to the image you would like to convert.")
    im = Image.open(sys.argv[1])
    im = im.convert("L")
    im_arr = np.asarray(im)
    edges = cv2.Canny(im_arr, 100, 200)
    edges = np.clip(edges, 0, 255)
    processed_im = Image.fromarray(edges.astype(np.uint8))
