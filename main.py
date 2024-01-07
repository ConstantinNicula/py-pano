import argparse
import loader
import time 
from matcher import Matcher
from bundler import Bundler,PoseVisualizer
from homography import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_src_folder", help="Path to the folder containing input images")
    parser.add_argument("out_folder", help="Path where result will be stored")
    args = parser.parse_args()

    imgs, imgs_exif = loader.load_batch(args.img_src_folder)
    
    st = time.time()
    # compute image matches
    matcher = Matcher()
    # match_data = matcher.match(imgs)
    match_data = matcher.match(imgs)
    print(f"Time to match: {time.time() - st}s") 

    bundler = Bundler()
    bundler.set_images(imgs)
    bundler.set_match_data(match_data)
    bundler.optimize(0)

    PoseVisualizer.display(bundler.pose_graph_nodes, imgs)
    
if __name__ == "__main__":
    main()