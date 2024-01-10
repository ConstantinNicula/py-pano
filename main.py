import argparse
import loader
import time 
from matcher import Matcher
from bundler import Bundler,PoseVisualizer
from homography import *
import cv2

def downscale_image(img: np.ndarray, target_dim: int)-> np.ndarray:
    h, w, _ = img.shape
    s = target_dim / max(h, w)
    if s > 1.0: 
        return img
    else: 
        dim = (int(w * s), int(h * s))
        return cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_src_folder", help="Path to the folder containing input images")
    parser.add_argument("out_folder", help="Path where result will be stored")
    args = parser.parse_args()

    imgs, imgs_exif = loader.load_batch(args.img_src_folder)
    st = time.time()

    # no need to use full scale images for pose estimation, reduce resolution
    mid_imgs = []    
    for img in imgs: 
        mid_imgs.append(downscale_image(img, 1024))

    # compute image matches
    matcher = Matcher()
    match_data = matcher.match(mid_imgs)
    print(f"Time to match: {time.time() - st}s") 

    bundler = Bundler()
    bundler.set_images(mid_imgs)
    bundler.set_match_data(match_data)
    bundler.optimize(0)

    PoseVisualizer.display(bundler.pose_graph_nodes, mid_imgs, [])
    
if __name__ == "__main__":
    main()