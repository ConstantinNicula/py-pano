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
        # Extents in horizontal direction
        self.min_w  = left
        self.max_w  = left + width

        # Extents in vertical direction
        self.min_h  = top
        self.max_h  = top + height

    def translate(self, offset_w: int, offset_h: int): 
        """
            Apply an offset in both width and height directions
        """
        # Shift in horizontal (width) direction
        self.min_w += offset_w
        self.max_w += offset_w
        
        # Shift in vertical (height) direction
        self.min_h += offset_h
        self.max_h += offset_h

    def relative_pos(self, x: int, y: int) -> tuple[int, int]:
        return (x - self.min_w, y - self.min_h) 

    @staticmethod
    def intersect(first, second) -> tuple[int, int, int, int] | None:  
        # Check intersect along width direction
        min_int_w = max(first.min_w, second.min_w)
        max_int_w = min(first.max_w, second.max_w)
        if max_int_w < min_int_w: return None 

        # Check intersect along height directions
        min_int_h = max(first.min_h, second.min_h)
        max_int_h = min(first.max_h, second.max_h)
        if max_int_h < min_int_h: return None

        # Get extents of intersection region 
        extent_w = max_int_w - min_int_w
        extent_h = max_int_h - min_int_h
        return (min_int_w, min_int_h, extent_w, extent_h)

    def test():
        return

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

        self.__compute_influence_mask()

    def __compute_influence_mask(self):

        # Compute weights for all images: 
        weights = []
        for img, pose in zip(self.imgs, self.poses):
            weights.append(self.__compute_warped_weights(img, pose))

        masks = []
        for i in range(len(self.imgs)):
            img = self.imgs[i]
            pose = self.poses[i]

            # Set initial mask 1 where warped weight > 0 
            mask = weights[i] > 0

            # Update influence mask
            for j in range(len(self.imgs)):
                if i == j: continue 

                # Compute bounds for intersection:
                reg = ImageBounds.intersect(self.warped_bounds[i], self.warped_bounds[j])
                if reg is None: continue            

                # If intersect exists update mask
                px, py, w, h = reg
                pxj, pyj = self.warped_bounds[j].relative_pos(px, py)
                pxi, pyi = self.warped_bounds[i].relative_pos(px, py)

                # Update mask in region 
                weight_j = weights[j][pyj:pyj+h, pxj:pxj+w]
                weight_i = weights[i][pyi:pyi+h, pxi:pxi+w]
                mask[pyi:pyi+h, pxi:pxi+w] &= weight_i > weight_j

            masks.append(mask.astype(np.float32))
            # mask_img = mask.astype(np.float32)
            # cv2.imshow(f"mask{i}", mask_img)

        return masks 

    # TODO: allow scaling
    def __compute_warped_weights(self, img: np.ndarray, pose: CameraPose) -> np.ndarray: 
        w, h = get_width_height(img) 

        # generate varying values from 0..1..0
        weights_x = 1.0 - np.abs(np.linspace(-1.0, 1.0, w)) 
        weights_y = 1.0 - np.abs(np.linspace(-1.0, 1.0, h))

        # take outer product to compute full weight matrix 
        weights_xy = np.outer(weights_y, weights_x)

        # convert weight to spherical coords 
        return self.__compute_warped_image(weights_xy, pose) 


    def __compute_warped_image(self, img: np.ndarray, pose: CameraPose) -> np.ndarray:
        invR = pose.get_inv_r_mat().astype(np.float32)
        K = pose.get_k_mat().astype(np.float32)
        _, warped_img = self.warper.warp(img, K, invR, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        return warped_img

    def compose(self) -> np.ndarray:
        # TODO: rework
        masks = self.__compute_influence_mask()
        for img, pose, bounds, mask in zip(self.imgs, self.poses, self.warped_bounds, masks):
            warped_img = self.__compute_warped_image(img, pose)
            self.__paste_image(warped_img, bounds, mask)
        return self.composite_image 

    # TODO: rework
    def __paste_image(self, warped_img: np.ndarray, bounds: ImageBounds, weight: list[np.ndarray]): 
        # dst = self.composite_image[bounds.min_h:bounds.max_h, bounds.min_w: bounds.max_w]
        h, w = weight.shape
        weight = np.reshape(weight, (h, w, 1))
        res = warped_img * weight 
        # res = np.where(warped_img != np.array([0, 0, 0]), warped_img, dst)
        self.composite_image[bounds.min_h:bounds.max_h, bounds.min_w: bounds.max_w] += res.astype(np.uint8)