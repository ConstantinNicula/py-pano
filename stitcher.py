import cv2
import numpy as np

from matcher import Matcher
from bundler import Bundler
from composer import Composer
from pose_debug import PoseVisualizer
from utils import *
import img_uitls

class Stitcher: 
    def __init__(self, img_dim_pose:int=1024):
        # Common properties 
        self.img_dim_pose = 1024

    def stitch(self, imgs: list[np.ndarray]) -> np.ndarray:
        # No need to use full scale images for pose estimation, reduce resolution
        scaled_imgs = []
        for img in imgs: 
            scaled_img, _= img_uitls.scale_image(img, self.img_dim_pose, cv2.INTER_AREA)
            scaled_imgs.append(scaled_img)

        # Compute image matches 
        # TODO: matching can yield multiple connected components, process each independently
        matcher = Matcher()
        match_data = matcher.match(scaled_imgs)

        # Perform pose estimation
        bundler = Bundler()
        bundler.set_images(scaled_imgs)
        bundler.set_match_data(match_data)
        poses = bundler.optimize(0)

        # Update poses 
        # TODO: maybe shift this inside Bundler 
        for i, img in enumerate(scaled_imgs):
            center = img_uitls.image_center(img)
            poses[i].set_camera_center(center)

        # Compose images
        # TODO: allow separate scaling for image compositing
        composer = Composer(scaled_imgs, poses)
        stitch_result = composer.compose()

        # show final result if requested
        if DEBUG_ENABLED():
            cv2.imshow(f"stitch_result", stitch_result)
            PoseVisualizer.display(bundler.pose_graph_nodes, scaled_imgs, [])

        return stitch_result