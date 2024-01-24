import argparse
import loader
import time 
import cv2
import os

from stitcher import Stitcher

def main():
    # Get command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("img_src_folder", help="Path to the folder containing input images")
    args = parser.parse_args()
    
    # Load input images
    imgs, _ = loader.load_batch(args.img_src_folder, load_exif=False)

    # Use stitching pipeline to generate panorama 
    stitcher = Stitcher()
    panorama = stitcher.stitch(imgs)

    # Store output 
    dest_folder = f"{args.img_src_folder}/out"
    if not os.path.exists(dest_folder): os.mkdir(dest_folder)
    cv2.imwrite(f"{dest_folder}/out.png", panorama)
    
if __name__ == "__main__":
    main()