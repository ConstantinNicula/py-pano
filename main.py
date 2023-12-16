import argparse
import loader
import time 
from matcher import Matcher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_src_folder", help="Path to the folder containing input images")
    parser.add_argument("out_folder", help="Path where result will be stored")
    args = parser.parse_args()

    imgs, imgs_exif = loader.load_batch(args.img_src_folder)
    matcher = Matcher()
    t1 = time.time()
    matcher.match(imgs)
    print(f"Elapsed type {time.time() - t1}s")
if __name__ == "__main__":
    main()