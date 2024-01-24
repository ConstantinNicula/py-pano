import argparse
import loader
import time 
import cv2
import os

from matcher import Matcher
from bundler import Bundler, CameraPose
from homography import *
from pose_debug import PoseVisualizer
from composer import Composer
from utils import *

import img_uitls
import so3

def downscale_image(img: np.ndarray, target_dim: int)-> np.ndarray:
    h, w, _ = img.shape
    s = target_dim / max(h, w)
    if s > 1.0: 
        return img
    else: 
        dim = (int(w * s), int(h * s))
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA) 


def pose_to_RK(w: float, h: float, pose: CameraPose): 
    f = pose.f
    R = so3.exp(pose.rot).astype(np.float32)
    K = np.array([[f, 0, w/2], 
                [0, f, h/2], 
                [0, 0, 1]]).astype(np.float32)
    return R, K

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_src_folder", help="Path to the folder containing input images")
    # parser.add_argument("out_folder", help="Path where result will be stored")
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

    # update poses
    for i, img, in enumerate(mid_imgs):
        center = img_uitls.image_center(img) 
        bundler.pose_graph_nodes[i].set_camera_center(center) 

    composer = Composer(mid_imgs, bundler.pose_graph_nodes)    
    panorama = composer.compose()
    
    dest_folder = f"{args.img_src_folder}/out"
    if not os.path.exists(dest_folder): os.mkdir(dest_folder)
    cv2.imwrite(f"{dest_folder}/out.png", panorama)
    
    # show final result if requested
    if DEBUG_ENABLED():
        cv2.imshow(f"panorama", panorama)
        PoseVisualizer.display(bundler.pose_graph_nodes, mid_imgs, [])

if __name__ == "__main__":
    main()