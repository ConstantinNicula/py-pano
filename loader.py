import os
import cv2
import numpy as np
import exifread
from utils import *

valid_img_extensions = ( ".jpg", ".jpeg", ".png", ".bmp")

def read_exif(img_path: str):
    with open(img_path, 'rb') as f: 
        return exifread.process_file(f)


def load_batch(img_src_folder: str, load_exif = True) -> tuple[list[np.ndarray], list[dict[str, any]]]:
    if not os.path.isdir(img_src_folder):
        raise Exception(f"Invalid folder path '{img_src_folder}'")

    batch_data = []
    batch_exif = []
    in_current_folder = os.listdir(img_src_folder)
    in_current_folder.sort()
    for img_name in in_current_folder:
        img_path = os.path.join(img_src_folder, img_name)
        if os.path.isfile(img_path) and img_path.lower().endswith(valid_img_extensions):
            batch_data.append(cv2.imread(img_path, cv2.IMREAD_COLOR))
            if load_exif: 
                batch_exif.append(read_exif(img_path))

            if DEBUG_ENABLED(): 
                print(f"Loaded image {img_path}")
        else: 
            if DEBUG_ENABLED(): 
                print(f"Skipping file {img_path}")

    return batch_data, batch_exif 