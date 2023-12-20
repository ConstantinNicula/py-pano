import argparse
import loader
import time 
from matcher import Matcher
from bundler import Bundler
from homography import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_src_folder", help="Path to the folder containing input images")
    parser.add_argument("out_folder", help="Path where result will be stored")
    args = parser.parse_args()

    imgs, imgs_exif = loader.load_batch(args.img_src_folder)
    
    # compute image matches
    matcher = Matcher()
    img_overlap_data = matcher.match(imgs)
    
    # create bundler
    bundler = Bundler()
    for i, img in enumerate(imgs):
        bundler.add_image(i, img)

    for i, j, H, kpts1, kpts2 in img_overlap_data:
        bundler.add_overlapping_points(i, j, kpts1, kpts2, H)
        break 

    bundler.optimize()

if __name__ == "__main__":
    main()