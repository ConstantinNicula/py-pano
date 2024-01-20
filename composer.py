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
        self.__compute_image_bounds() 

    def __compute_scale(self) -> float: 
        cam_focal_lengths = [pose.f for pose in self.poses] 
        return np.median(cam_focal_lengths)

    # NOTE: may not work if images are flipped 
    def __compute_image_bounds(self) -> list[tuple[int]]:
        min_left, min_top = np.inf, np.inf # top left corner of warped ROI
        max_right, max_bottom = -np.inf, -np.inf # bottom right corner of warped ROI 

        for img, pose in zip(self.imgs, self.poses):
            w, h = get_width_height(img)

            # Compute warped roi
            invR = pose.get_inv_r_mat().astype(np.float32)
            K = pose.get_k_mat().astype(np.float32)
            left, top, width_warp, height_warp = self.warper.warpRoi((w, h), K, invR)
            self.warped_bounds.append(ImageBounds(left, top, width_warp, height_warp))

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

    def compose(self, num_bands=2) -> np.ndarray:
        # TODO: rework
        all_block_bands, all_block_weights = self.__compute_blend_bands(num_bands=num_bands)
        
        # create output for final image
        img_shape = self.composite_image.shape 
        full_acc = np.zeros(img_shape, dtype=np.float32)

        for k in range(num_bands): 
            # create accumulator images for intermediates 
            band_acc = np.zeros(img_shape, dtype=np.float32)
            weight_acc = np.full((img_shape[0], img_shape[1]), 1e-6, dtype=np.float32)  

            for bounds, block_bands, block_weights in zip(self.warped_bounds, all_block_bands, all_block_weights):
                # write image data
                self.__paste_block(block_bands[k], bounds, band_acc, weight=block_weights[k])
                self.__paste_block(block_weights[k], bounds, weight_acc)
            # cv2.imshow(f"weight_acc{k}", weight_acc/np.max(weight_acc))
            # cv2.imshow(f"band_acc{k}", band_acc / 255)
            band_acc /= np.expand_dims(weight_acc, axis =-1)

            # cv2.imshow(f"norm_band_acc{k}", band_acc / 255)
            full_acc += band_acc  

        self.composite_image = np.clip(full_acc, 0, 255).astype(np.uint8)

        # for img, pose, bounds, mask in zip(self.imgs, self.poses, self.warped_bounds, influence_mask):
        #     warped_img = self.__compute_warped_image(img, pose)
        #     self.__compute_blend_bands(img, mask)
        #     self.__paste_image(warped_img, bounds, mask)
        return self.composite_image 

    def __compute_blend_bands(self, sigma: float=5, num_bands: int=3) -> tuple[list[np.ndarray], list[np.ndarray]]: 
        # TODO: Convolutions are not calculated correctly (should switch to image coords to avoid errors)
        # Compute requirements  
        warp_imgs = [self.__compute_warped_image(img, pose) for img, pose in zip(self.imgs, self.poses)]  
        influence_masks = self.__compute_influence_masks()         

        # Storage for aggregate bands
        all_bands = []
        all_weights = []

        # Loop through images and calculate bands 
        for warp_img, inf_mask in zip(warp_imgs, influence_masks):  
            # Preallocate for speed
            I = np.zeros((num_bands, *warp_img.shape), dtype=np.float32) 
            B = np.zeros((num_bands, *warp_img.shape), dtype=np.float32)
            W = np.zeros((num_bands, *inf_mask.shape), dtype=np.float32)

            # Storage for blurred images and bands 
            # I[0] - warped_img; B[k] = I[k] - I[K+1]
            I[0] = warp_img.astype(np.float32)

            # Compute image high-pass bands
            # B[k] includes wavelengths in range [0, k*sigma]
            for k in range(0, num_bands-1):
                # Compute  I[k+1] as I[k] * g_sigma_k
                I[k+1] = blur_image(I[k], sigma * np.sqrt(2*k+1))  
                # Compute band B[k] as I[k+1] - I[k]
                B[k] = I[k] - I[k+1]            

            # Add final band for interval [num_bands*sigma, inf]  
            B[-1] = I[-1]

            # Storage for band weights 
            # W[k] - corresponds to band B[k]
            W[0] = blur_image(inf_mask, sigma) 
            for k in range(0, num_bands - 1): 
                # Compute W[k+1] as W[k] * g_sigma_k 
                W[k+1] = blur_image(W[k], sigma * np.sqrt(2*(k+1) + 1))

            # Save 
            all_bands.append(B)
            all_weights.append(W)

        return all_bands, all_weights 
 
    def __compute_influence_masks(self) -> list[np.ndarray]:
        influence_masks = []
        warped_weights = self.__compute_warped_weights() 

        # Compute influence masks for all images:
        num_imgs = len(self.imgs)
        for i in range(num_imgs):
            # Set initial mask 1 where warped weight > 0 
            mask = warped_weights[i] > 0

            # Update influence mask
            for j in range(num_imgs):
                if i == j: continue 

                # Compute bounds for intersection:
                reg = ImageBounds.intersect(self.warped_bounds[i], self.warped_bounds[j])
                if reg is None: continue            

                # If intersect exists update mask
                px, py, w, h = reg
                pxj, pyj = self.warped_bounds[j].relative_pos(px, py)
                pxi, pyi = self.warped_bounds[i].relative_pos(px, py)

                # Update mask in region 
                weight_j = warped_weights[j][pyj:pyj+h, pxj:pxj+w]
                weight_i = warped_weights[i][pyi:pyi+h, pxi:pxi+w]
                mask[pyi:pyi+h, pxi:pxi+w] &= weight_i > weight_j # abuse of bitwise op?? 
            influence_masks.append(mask.astype(np.float32))

        return influence_masks

    # TODO: allow scaling
    def __compute_warped_weights(self) -> list[np.ndarray]: 
        warped_weights = []
        for img, pose in zip(self.imgs, self.poses):
            w, h = get_width_height(img) 

            # Generate varying values from 0..1..0
            weights_x = 1.0 - np.abs(np.linspace(-1.0, 1.0, w)) 
            weights_y = 1.0 - np.abs(np.linspace(-1.0, 1.0, h))

            # Take outer product to compute full weight matrix 
            weights_xy = np.outer(weights_y, weights_x)

            # Convert weight to spherical coords 
            warped_weights.append(self.__compute_warped_image(weights_xy, pose)) 
        return warped_weights

 
    def __compute_warped_image(self, img: np.ndarray, pose: CameraPose) -> np.ndarray:
        invR = pose.get_inv_r_mat().astype(np.float32)
        K = pose.get_k_mat().astype(np.float32)
        _, warped_img = self.warper.warp(img, K, invR, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
        return warped_img

    # TODO: rework
    def __paste_block(self, img_block: np.ndarray, bounds: ImageBounds, out_img: np.ndarray, weight: np.ndarray|None = None): 
        res = img_block
        if weight is not None: res *= np.expand_dims(weight, axis=-1) 
        out_img[bounds.min_h:bounds.max_h, bounds.min_w: bounds.max_w] += res