import cv2 
import time
import numpy as np
from utils import * 
from homography import *

class Matcher: 
    def __init__(self, kpts_per_img:int = 2000, reproj_err:float=3): 
        # Params described https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
        self.kp_detector = cv2.ORB.create(nfeatures=kpts_per_img, scaleFactor=1.4)
        self.kp_matcher = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)
        self.repoj_err = reproj_err
        
    def match(self, imgs: list[np.ndarray]): 
        """
            Detect overlapping images.
            Returns dict [(i,j)] = (H, kpts1, kpts2)
                - i, j - indices of overlapping images
                - H - estimated homography
                - kpts1, ktps2 - ndarrays of inliers matches
        """
        # preprocess all images
        img_descriptors = []
        img_keypoints = []
        for img in imgs: 
            keypoints, descriptors = self.__extract_keypoints(img)
            img_descriptors.append(descriptors)
            img_keypoints.append(keypoints)

        # Quadratic matching between all images. 
        # An approximate solution can be obtained in n*log(n) but it's more difficult to implement
        img_overlap_data = {}
        for i in range(len(imgs) - 1):
            for j in range(i + 1, len(imgs)):
                # get ordered pairs of matching keypoints
                m_kpts_i, m_kpts_j = self.__match_keypoints(img_keypoints[i], img_descriptors[i], img_keypoints[j], img_descriptors[j])

                # filter using geometric check
                overlap_data = self.__check_valid_homography(m_kpts_i, m_kpts_j, self.repoj_err)
                if overlap_data:
                    if DEBUG_ENABLED(): print(f"Found valid match {i} -> {j}") 
                    img_overlap_data[(i, j)] = overlap_data 
        return img_overlap_data
    
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

    def __match_keypoints(self, kpts1: np.ndarray, desc1: np.ndarray, kpts2: np.ndarray, desc2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
            Matches two sets of ORB descriptors 
            Returns tuple (m_kpts1, m_kpts2) of ndarrays which specify matching points m_kpts1[0] <-> m_kpts2[0] 
        """
        # desc1 - query image, desc2 - train image
        kpt_matches = self.kp_matcher.knnMatch(desc1, desc2, k=2)
        good_kpts1, good_kpts2 = [], []
        for m in kpt_matches:
            if len(m) == 2 and m[0].distance < 0.7 * m[1].distance:
                good_kpts1.append(m[0].queryIdx)
                good_kpts2.append(m[0].trainIdx)

        # filter relevant keypoints
        return kpts1[good_kpts1], kpts2[good_kpts2]


    def __check_valid_homography(self, kpts1: np.ndarray, kpts2: np.ndarray, reproj_err = 5,  alpha = 8.0, beta = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Determine if there is a valid homography between kpts1 and kpts2, based on provided matches 
            Returns a tuple containing (H, kpts1, kpt2)
                - H - calculated homography matrix 
                - kpts1, kpts2 - pairs of valid matches (with reprojection error in specified bounds) 
        """
        if len(kpts1) != len(kpts2) or len(kpts1) < 4:
            return None 

        H, inliner_mask = estimate_homography_ransac(kpts1, kpts2, reproj_err) 
        nf, ni = len(kpts1), len(inliner_mask)
        valid = ni > alpha + beta * nf

        if DEBUG_ENABLED(): 
            print(f"total matches {nf}, valid matches {ni}")
        return (H, kpts1[inliner_mask], kpts2[inliner_mask]) if valid else None

