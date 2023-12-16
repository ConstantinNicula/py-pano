import cv2 
import time
import numpy as np
from utils import * 
from homography import *

class Matcher: 
    def __init__(self, kpts_per_img:int = 2000): 
        # Params described https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
        self.kp_detector = cv2.ORB.create(nfeatures=kpts_per_img, scaleFactor=1.4)
        self.kp_matcher = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)

        # Contains concatenated data
        self.img_descriptors = []
        self.img_keypoints = [] 
        
        # Contains matches between tuples which describe image overlaps:  
        # hash[(i,j)] = (H, inlier_matches)
        # i, j - indices of overlapping images
        # H - estimated homography
        # inlier_matches - ndarray of inliers matches [(k, p) ...] 
        # keypoint k in image i matches with keypoint p in image j  
        self.img_overlap_data = {}
        
        # Pairs (i, j) of overlapping images
        self.img_connectivity = []

    def get_overlapping_pairs(self):
        """
            Returns a list of tuples containing valid image overlaps
        """ 
        return self.img_connectivity

    def get_overlap_data(self, img1_id: int, img2_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None: 
        """
            Returns tuple (H, kp1, kp2) containing overlap data between img1 and img2 or None if no overlap exists
            H - homography from img1 to img2
            kp1 - list of keypoints in img1 
            kp2 - list of keypoints in img2
        """
        if img2_id > img1_id:
            img1_id, img2_id = img2_id, img1_id

        if (img1_id, img2_id) not in self.img_overlap_data:
            return None

        H, matches = self.img_overlap_data[(img1_id, img2_id)]  
        img1_kp = self.img_keypoints[img1_id][matches[:, 0]]
        img2_kp = self.img_keypoints[img2_id][matches[:, 1]]
        return (H, img1_kp, img2_kp)

    def match(self, imgs: list[np.ndarray], m:int = 6): 
        """
            Detect overlapping images.
        """
        # preprocess all images
        for img in imgs: 
            keypoints, descriptors = self.__extract_keypoints(img)
            self.img_descriptors.append(descriptors)
            self.img_keypoints.append(keypoints)

        # Quadratic matching between all images. 
        # An approximate solution can be obtained in n*log(n) but it's more difficult to implement
        for i in range(len(imgs) - 1):
            for j in range(i + 1, len(imgs)):
                kp_raw_matches = self.__match_keypoints(self.img_descriptors[i], self.img_descriptors[j])

                # filter using geometric check
                overlap_data = self.__is_valid_overlap(self.img_keypoints[i], self.img_keypoints[j], kp_raw_matches)
                if overlap_data:
                    if DEBUG_ENABLED(): print(f"Found valid match {i} -> {j}") 
                    self.img_overlap_data[(i, j)] = overlap_data 
        
        self.img_connectivity = self.img_overlap_data.keys()

    def __extract_keypoints(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
        """
            Extracts ORB keypoints and descriptors
            Returns tuple of two ndarrays:
                - kps.shape = (n, 3) - homogenous coordinates
                - descriptor.shape = (n, 32) 
        """
        # convert image to gray scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv_kpts, desc = self.kp_detector.detectAndCompute(img_gray, None) 
        kpts = convert_to_homogenous(np.array([kp.pt for kp in cv_kpts]))
        return kpts, desc 

    def __match_keypoints(self, desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        """
            Matches two sets of ORB descriptors 
            Returns ndarray of shape (n, 2):
                - match[:, 0] -> index of train kp
                - match[:, 1] - > index of query kp
        """
        # desc1 - query image, desc2 - train image
        kpt_matches = self.kp_matcher.knnMatch(desc1, desc2, k=2)
        good_kpt_matches = []

        for m in kpt_matches:
            if len(m) == 2 and m[0].distance < 0.7 * m[1].distance:
                good_kpt_matches.append(m[0])
        return np.array([(m.queryIdx, m.trainIdx) for m in good_kpt_matches])


    def __is_valid_overlap(self, kpts1: np.ndarray, kpts2: np.ndarray, matches: np.ndarray, alpha = 8.0, beta = 0.3):
        """
            Determine if there is a valid homography between kpts1 and kpts2, based on provided matches 
            Returns a tuple containing (valid_overlap, est_homography, kpts_inliers)
                - est_homography - calculated homography matrix 
                - matches - corresponding to inliers 
        """
        if len(matches) < 4:
            return None 

        H, inliner_mask = estimate_homography_ransac(kpts1, kpts2, matches) 
        nf, ni = len(matches), len(inliner_mask)
        valid = ni > alpha + beta * nf
        if DEBUG_ENABLED(): 
            print(f"total matches {nf}, valid matches {ni}")
        return (H, matches[inliner_mask]) if valid else None

