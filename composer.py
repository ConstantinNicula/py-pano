import cv2 
import numpy as np
import so3

from bundler import CameraPose
from img_uitls import *

class ImageBounds: 
    """ 
        Class for keeping track of warped image extents 
    """
    def __init__(self, left: int, top: int, width: int, height: int): 
        self.min_w  = left
        self.max_w  = left + width

        self.min_h  = top
        self.max_h  = top + height

    def translate(self, offset_w: int, offset_h: int): 
        """
            Apply an offset in both width and height directions
        """
        self.min_w += offset_w
        self.max_w += offset_w

        self.min_h += offset_h
        self.max_h += offset_h

class Composer:
    def __init__(self, imgs: list[np.ndarray], poses: list[CameraPose]):
        self.imgs = imgs
        self.poses = poses

        # Create cv2 warper  
        self.scale = self.__compute_scale()
        self.warper = cv2.PyRotationWarper("spherical", self.scale)
        
        # Storage for internals  
        self.warped_bounds: list[ImageBounds] = [] 
        self.composite_image: np.ndarray = None 
        self.__compute_bounds() 

    def __compute_scale(self) -> float: 
        cam_focal_lengths = [pose.f for pose in self.poses] 
        return np.median(cam_focal_lengths)

    # TO DO: may not work if images are flipped 
    def __compute_bounds(self) -> list[tuple[int]]:
        min_left, min_top = np.inf, np.inf # top left corner of warped ROI
        max_right, max_bottom = -np.inf, -np.inf # bottom right corner of warped ROI 

        for img, pose in zip(self.imgs, self.poses):
            w, h = get_width_height(img)

            # Compute warped roi
            invR = pose.get_inv_r_mat().astype(np.float32)
            K = pose.get_k_mat().astype(np.float32)
            left, top, width_warp, height_warp = self.warper.warpRoi((w, h), K, invR)
            self.warped_bounds.append(ImageBounds(left, top, width_warp, height_warp))
            print(left, top, width_warp, height_warp)
            # Update extents in horizontal direction
            min_left = min(left, min_left) 
            max_right = max(left + width_warp, max_right)

            # Update extents in vertical direction
            min_top = min(top, min_top)
            max_bottom = max(top + height_warp, max_bottom)

        # Loop again and compute real bounds
        for bound in self.warped_bounds:
            bound.translate(-min_left, -min_top)

        # Create empty composite image          
        full_width = max_right - min_left + 1
        full_height = max_bottom - min_top + 1 
        self.composite_image = np.zeros((full_height, full_width, 3), dtype=np.uint8)

    def compose(self) -> np.ndarray:
        # TODO: rework
        for img, pose, bounds in zip(self.imgs, self.poses, self.warped_bounds):
            invR = pose.get_inv_r_mat().astype(np.float32)
            K = pose.get_k_mat().astype(np.float32)
            _, warped_img = self.warper.warp(img, K, invR, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT) 
            self.__paste_image(warped_img, bounds)
        return self.composite_image 

    # TODO: rework
    def __paste_image(self, warped_img: np.ndarray, bounds: ImageBounds): 
        dst = self.composite_image[bounds.min_h:bounds.max_h, bounds.min_w: bounds.max_w]
        res = np.where(warped_img != np.array([0, 0, 0]), warped_img, dst)
        self.composite_image[bounds.min_h:bounds.max_h, bounds.min_w: bounds.max_w] = res