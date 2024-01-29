import sys
import cv2 
import numpy as np
from dataclasses import dataclass

from debug_utils import * 
from homography import *


@dataclass
class MatchData:
    """
        Class for keeping track of overlap information between two images:
            - img1_id: id of img1  
            - img2_id: id of img2 
            - img1_kpts: keypoints in img1 (origin at image top left)
            - img2_kpts: keypoints in img2 (origin at image top left)
            - H: homography from img1 to img2
    """
    img1_id: int 
    img2_id: int
    img1_kpts: np.ndarray
    img2_kpts: np.ndarray
    H: np.ndarray

class Matcher: 
    def __init__(self, kpts_per_img:int = 1500, reproj_err:float=3): 
        self.kp_detector = cv2.SIFT.create(nfeatures=kpts_per_img,
                                           nOctaveLayers=3,
                                           contrastThreshold=0.09)
        self.kp_matcher = cv2.BFMatcher.create(normType=cv2.NORM_L2)
        self.repoj_err = reproj_err
        
    def match(self, imgs: list[np.ndarray]) -> list[MatchData]: 
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
        img_overlap_data = [] 
        for i in range(len(imgs) - 1):
            for j in range(i + 1, len(imgs)):
                # get ordered pairs of matching keypoints
                m_kpts_i, m_kpts_j = self.__match_keypoints(img_keypoints[i], img_descriptors[i], img_keypoints[j], img_descriptors[j])

                # filter using geometric check
                res = self.__check_valid_homography(imgs[i], imgs[j], m_kpts_i, m_kpts_j, self.repoj_err)
                if res:
                    if DEBUG_ENABLED(): print(f"Found valid match {i} -> {j}") 
                    H, kpts1, kpts2 = res
                    img_overlap_data.append(MatchData(i, j, kpts1, kpts2, H)) 
        return img_overlap_data
    
    def __extract_keypoints(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
        """
            Extracts ORB keypoints and descriptors
            Returns tuple of two ndarrays:
                - kpts: keypoints in homogenous coordinates
                - descriptor: descriptors of keypoints = (n, 32) 
        """
        # convert image to gray scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv_kpts, desc = self.kp_detector.detectAndCompute(img_gray, None) 
        kpts = convert_to_homogenous(np.array([kp.pt for kp in cv_kpts]))
        return kpts, desc 

    def __match_keypoints(self, kpts1: np.ndarray, desc1: np.ndarray, kpts2: np.ndarray, desc2: np.ndarray, n_repeated=3) -> tuple[None|np.ndarray, None|np.ndarray]:
        """
            Matches two sets of ORB descriptors 
            Returns tuple (m_kpts1, m_kpts2) of ndarrays which specify matching points m_kpts1[0] <-> m_kpts2[0] 
        """
        # desc1 - query image, desc2 - train image
        kpt_matches = self.kp_matcher.knnMatch(desc1, desc2, k=2)
        
        train_to_query: dict[int, list] = {i:[] for i in range(len(desc2)) }
        for m in kpt_matches:
            if len(m) == 2 and m[0].distance < 0.8 * m[1].distance:
                train_to_query[m[0].trainIdx].append((m[0].queryIdx, m[0].distance))
   
        good_kpts1, good_kpts2 = [], []
        for train_id in train_to_query:
            # skip non matched
            if len(train_to_query[train_id]) == 0: 
                continue 

            # sort multiple matches by distance
            train_to_query[train_id].sort(key = lambda m: m[1]) 
        
            # currently taking only the two best
            for i in range(min(len(train_to_query[train_id]), n_repeated)):
                query_id, _ = train_to_query[train_id][i]
                good_kpts2.append(train_id)
                good_kpts1.append(query_id)

        # return filtered points
        return kpts1[good_kpts1], kpts2[good_kpts2]


    def __check_valid_homography(self, img1:np.ndarray, img2: np.ndarray, 
                                 kpts1: np.ndarray, kpts2: np.ndarray, 
                                 reproj_err = 5,  alpha = 8.0, beta = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Determine if there is a valid homography between kpts1 and kpts2, based on provided matches 
            Returns a tuple containing (H, kpts1, kpt2)
                - H - calculated homography matrix 
                - kpts1, kpts2 - pairs of valid matches (with reprojection error in specified bounds) 
        """
        if len(kpts1) != len(kpts2) or len(kpts1) < 4:
            return None 

        # compute homography
        H, inliner_mask = estimate_homography_ransac(kpts1, kpts2, reproj_err)
        
        # can't possible pass the geometric test, early exit
        if np.count_nonzero(inliner_mask) <= alpha: return None

        # find matches in overlap region        
        area_img2, poly_img2 = self.__compute_overlap_poly(img1.shape, img2.shape, H)
        area_img1, poly_img1 = self.__compute_overlap_poly(img2.shape, img1.shape, np.linalg.inv(H))

        if poly_img1 is None or poly_img2 is None: return None
        
        # number of features and number of inliers in overlap region  
        nf, ni = 0, 0
        for i in range(len(kpts1)):
            img2_valid = cv2.pointPolygonTest(poly_img2, kpts2[i, 0:2], False) >= 0
            img1_valid = cv2.pointPolygonTest(poly_img1, kpts1[i, 0:2], False) >= 0           

            if img1_valid and img2_valid: 
                nf += 1
                ni += inliner_mask[i]

        # perform validation check
        valid = ni > alpha + beta * nf

        if DEBUG_ENABLED() == 2 and valid:
            image1 = img1.copy()
            image2 = img2.copy()

            for i in range(len(kpts1)):
                color = (0, 0, 255) if not inliner_mask[i] else (0, 255, 0)
                image1 = cv2.circle(image1, (int(kpts1[i, 0]), int(kpts1[i, 1])), radius=2, color=color, thickness=-1)
                image2 = cv2.circle(image2, (int(kpts2[i, 0]), int(kpts2[i, 1])), radius=2, color=color, thickness=-1)

            image1 = cv2.polylines(image1, [poly_img1.astype(np.int32)], True, (255, 0, 0), 2)
            image2 = cv2.polylines(image2, [poly_img2.astype(np.int32)], True, (0, 255, 0), 2)
            
            cv2.imshow("overlaps", np.concatenate((image1, image2), axis=1))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if DEBUG_ENABLED(): 
            print(f"total matches {nf}, valid matches {ni}")
        return (H, kpts1[inliner_mask], kpts2[inliner_mask]) if valid else None


    def __compute_overlap_poly(self, src_img_size: np.ndarray, dst_img_size: np.ndarray, H: np.ndarray) -> tuple[float, np.ndarray]:
        pts1 = self.__get_img_boundary_pts(dst_img_size)[:, 0:2].astype(np.float32)
        pts2 = transform_points(H, self.__get_img_boundary_pts(src_img_size))[:, 0:2].astype(np.float32)

        # intersect convex convex only accepts points with coords > 0, shift everything
        shift = -min(np.min(pts1), np.min(pts2))
        pts1, pts2 = pts1 + shift, pts2 + shift
        area, intersect = cv2.intersectConvexConvex(pts1, pts2, handleNested=True)
        if intersect is not None:
            intersect -= shift
            intersect = np.reshape(intersect, (-1, 2))
        return area, intersect 

    def __get_img_boundary_pts(self, img_size: tuple[int]) -> np.ndarray: 
        h, w = img_size[0], img_size[1] 
        return np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])