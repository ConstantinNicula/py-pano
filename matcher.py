import cv2 
import time
import numpy as np
import numpy.linalg
import itertools

class Matcher: 
    def __init__(self, kpts_per_img:int = 2000): 
        # Params described https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
        self.kp_detector = cv2.ORB.create(nfeatures=kpts_per_img, scaleFactor=1.4)
        self.kp_matcher = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)

        # Contains concatenated data
        self.img_descriptors = []
        self.img_keypoints = [] 
        
        # Contains matches between keypoints in img_i and img_k  where k > i 
        # Results are filtered and contain matches only for valid overlaps
        self.img_matches = [] 

    """
        Detect overlapping images.
    """
    def match(self, imgs: list[np.ndarray], m:int = 6): 
        # preprocess all images
        for img in imgs: 
            keypoints, descriptors = self.__get_keypoints(img)
            self.img_descriptors.append(descriptors)
            self.img_keypoints.append(keypoints)

        # Quadratic matching between all images. 
        # An approximate solution can be obtained in n*log(n) but it's more difficult to implements 
        self.img_matches = []  
        for i in range(len(imgs) - 1):
            for j in range(i + 1, len(imgs)):
                kp_raw_matches = self.__match_keypoints(self.img_descriptors[i], self.img_descriptors[j])

                # filter using geometric check
                valid, H, inliner_kp_matches = self.__is_valid_overlap(self.img_keypoints[i], self.img_keypoints[j], kp_raw_matches)
                if valid:
                    print(f"Found valid match {i} -> {j}") 
                    self.img_matches.append((i, j, inliner_kp_matches))                    

        return self.img_matches

    """
        Extracts ORB keypoints and descriptors
        Returns tuple of two ndarrays:
            - kps.shape = (n, 2)
            - descriptor.shape = (n, 32) 
    """
    def __get_keypoints(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
        # convert image to gray scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv_kpts, desc = self.kp_detector.detectAndCompute(img_gray, None) 
        return np.array([kp.pt for kp in cv_kpts]), desc 

    """
        Matches two sets of ORB descriptors 
        Returns ndarray of shape (n, 2):
            - match[:, 0] -> index of train kp
            - match[:, 1] - > index of query kp
    """
    def __match_keypoints(self, desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        # desc1 - query image, desc2 - train image
        kpt_matches = self.kp_matcher.knnMatch(desc1, desc2, k=2)
        good_kpt_matches = []
        for m in kpt_matches:
            if len(m) == 2 and m[0].distance < 0.7 * m[1].distance:
                good_kpt_matches.append(m[0])
        return np.array([(m.queryIdx, m.trainIdx) for m in good_kpt_matches])

    """
        Determine if there is a valid homography between kpts1 and kpts2, based on provided matches 
        Returns a tuple containing (valid_overlap, est_homography, kpts_inliers)
            - valid_overlap - bool indicating if there is a valid homography between the provided points
            - est_homography - calculated homography matrix 
            - kpts_inliers - filtered keypoints 
    """

    def __is_valid_overlap(self, kpts1: np.ndarray, kpts2: np.ndarray, matches: np.ndarray, alpha = 8.0, beta = 0.3):
        if len(matches) < 4:
            return False, None, None

        H, inliner_mask = self.__estimate_homography_ransac(kpts1, kpts2, matches) 
        nf, ni = len(matches), len(inliner_mask)
        valid = ni > alpha + beta * nf
        return valid, H, matches[inliner_mask]

    def __estimate_homography_ransac(self, kpts1: np.ndarray, kpts2: np.ndarray, matches: np.ndarray, reproj_err=5, max_iters=500) -> tuple[np.ndarray, np.ndarray]:
        # Implementation based on: 
        # https://engineering.purdue.edu/kak/courses-i-teach/ECE661.08/solution/hw4_s1.pdf

        H = None
        inlier_mask = None
        max_num_inliers = 0
        err_std = np.Inf

        # reorder points according to matches 
        kpts1_h = self.__to_homogenous(kpts1[matches[:, 0]])
        kpts2_h = self.__to_homogenous(kpts2[matches[:, 1]]) 
         
        i, N = 0, max_iters 
        while i < N:
            # extract 4 potential inliers at random
            sel_pairs = np.random.choice(len(matches), 4, replace=False)
            sel_kpts1 = kpts1_h[sel_pairs]
            sel_kpts2 = kpts2_h[sel_pairs]

            # estimate homography from 4 pairs 
            Hi = self.__estimate_homography(sel_kpts1, sel_kpts2) 

            # calculate the number of inliers
            cur_num_inliers, curr_std, curr_inlier_mask = self.__find_inliers(Hi, kpts1_h, kpts2_h, reproj_err) 
            
            # update model if necessary
            if cur_num_inliers > max_num_inliers or (cur_num_inliers == max_num_inliers and curr_std < err_std):
                # store best model
                H = Hi
                max_num_inliers, err_std, inlier_mask = cur_num_inliers, curr_std, curr_inlier_mask
                # print(f"Iter {i} nr_matches:{cur_num_inliers}, err_std:{curr_std}")
            
            # update N            
            e = 1 - cur_num_inliers / len(matches) + 1e-9
            N = min(max_iters, int(np.log(1-0.99)/ np.log(1-(1-e)**4)))

            i += 1
        return H, inlier_mask

    def __to_homogenous(self, kpts: np.ndarray): 
        ones_col = np.ones((len(kpts), 1))
        return np.hstack((kpts, ones_col))

    def __estimate_homography(self, kpts1: np.ndarray, kpts2: np.ndarray) -> np.ndarray:
        assert len(kpts1) == len(kpts2) and len(kpts1) == 4

        A = np.zeros((8, 9))
        for i in range(4):
            pta, ptb = kpts1[i], kpts2[i]
            
            # xi, yi, 1, 0, 0, 0, -xi' * xi, -xi' * yi, -xi' 
            A[2*i, 0:3] = pta 
            A[2*i, 6:9] = -ptb[0] * pta 

            # 0, 0, 0, xi, yi, 1, -yi' * xi, -yi' *yi, -yi' 
            A[2*i+1, 3:6] = pta 
            A[2*i+1, 6:9] = -ptb[1] * pta 

        # Use SVD to find null space 
        AtA = A.T @ A 
        U, S, V = np.linalg.svd(AtA)

        # Extract solution from last column (normalized)
        return np.reshape(V[-1, :], (3, 3)) / V[-1,-1] 
    
    def __find_inliers(self, H: np.ndarray, kpts1: np.ndarray, kpts2: np.ndarray, reproj_err: float) -> tuple[int, float, np.ndarray]: 
        # compute transformed points
        pt_est = (H @ kpts1.T).T
        pt_est = (pt_est.T / pt_est[:, 2]).T

        # compute estimation error
        err = kpts2 - pt_est
        norm_err = np.linalg.norm(err, axis=1)

        # extract inliers and the corresponding errors
        inlier_mask = np.flatnonzero(norm_err < reproj_err)
        errors = norm_err[inlier_mask]

        return len(inlier_mask), np.std(errors), inlier_mask

    def __utils_resize(self, img: np.ndarray, scale = 0.5):
        new_dims = (int(img.shape[1] * scale), int(img.shape[0] * scale)) 
        return cv2.resize(img, new_dims)

import loader
if __name__ == "__main__":
    imgs, exif = loader.load_batch("./input_imgs/field")
    # imgs, exif = loader.load_batch("./input_imgs/grandesjorasses")
    # imgs, exif = loader.load_batch("./input_imgs/apron")
    matcher = Matcher()
    t1 = time.time()
    matcher.match(imgs)
    print(f"Delta time {time.time() - t1}")
    #matcher.test_extract(imgs[0])
    #res = matcher.test_match(imgs[0], imgs[2])
