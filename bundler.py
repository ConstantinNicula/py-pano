import numpy as np
from collections import deque
from scipy.optimize import least_squares
from dataclasses import dataclass
from matcher import MatchData

import so3
from homography import *
from utils import *

@dataclass
class OverlapData:
    """
        Class for keeping track of overlap information between two images:
            - src_kpts: keypoints in source img (origin at image center)
            - dst_kpts: keypoints in destination img (origin at image center)
            - H: homography from source to destination 
    """
    src_kpts: np.ndarray
    dst_kpts: np.ndarray
    H: np.ndarray

class CameraPose: 
    """
        Class for keeping track of camera params: 
            - Source image
            - Intrinsic: camera focal length estimate 
            - Extrinsic: Camera rotation estimate 
    """
    def __init__(self, f: float=1.0, rot: np.ndarray = np.zeros(3)):
        # camera intrinsics parameters
        self.f = f
        # camera extrinsic parameters
        self.rot = rot 

    def get_cam_mat(self):
        return self.get_k_mat() @ so3.exp(self.rot) 

    def get_inv_cam_mat(self): 
       return so3.exp(self.rot).T @ self.get_inv_k_mat() 

    def get_k_mat(self): 
        return np.array([[self.f, 0, 0], 
                        [0, self.f, 0], 
                        [0, 0, 1]])

    def get_inv_k_mat(self):
        return np.array([[1/self.f, 0, 0], 
                         [0, 1/self.f, 0], 
                         [0, 0, 1]])
    
    def __repr__(self):
        return f"f: {self.f} rot: {self.rot}"

    @staticmethod
    def rot_from_homography(pose1, pose2, H21: np.ndarray) -> np.ndarray:
        # find best pose2.rot that best explains H21 (homography from 1 to 2)
        # H21 = K2 * R2 * R1^-1 * K1^-1 
        # R2 * R1^-1 = K2^-1 * H21 * K1
        # R21 = K2^-1 * H21 * K1
        R21 = pose2.get_inv_k_mat() @ H21 @ pose1.get_k_mat()
        R21 /= np.linalg.det(R21)
        w21 = so3.log(R21)
        print(f"relative rot {w21=:}")
        return so3.log( so3.exp(w21) @ so3.exp(pose1.rot)) 

class Bundler: 
    def __init__(self, imgs: list[np.ndarray]|None = None, img_matches: list[MatchData]|None = None):
        # storage for images
        self.imgs = [] 
        
        # camera poses are graph nodes
        self.pose_graph_nodes: list[CameraPose] = []

        # adjacency (overlap) data 
        self.pose_graph_edges: dict[tuple[int, int], OverlapData]= {} 

        # store provided data
        if imgs: self.set_images(imgs)
        if img_matches: self.set_match_data(img_matches)
        
        # ids nodes which are currently being processed
        self.active_node_ids: list[int] = []
        self.active_edges: list[tuple(int, int, OverlapData)] = [] 

    def set_images(self, imgs: list[np.ndarray]):
        assert imgs is not None
        self.imgs = imgs
        self.pose_graph_nodes = [CameraPose() for _ in imgs]

    def set_match_data(self, img_matches: list[MatchData]):
        assert img_matches is not None
        self.img_matches = {} 

        for mdata in img_matches: 
            img1_id, img2_id = mdata.img1_id, mdata.img2_id
            img1_kpts, img2_kpts, H = mdata.img1_kpts, mdata.img2_kpts, mdata.H

            # recenter points and compute new homography 
            img1_kpts, img2_kpts, H = self.__recenter_keypoints(self.imgs[img1_id], self.imgs[img2_id], 
                                                                img1_kpts, img2_kpts, H) 

            # Store adjacency info and measurement data 
            self.pose_graph_edges[(img1_id, img2_id)] = OverlapData(img1_kpts, img2_kpts, H)
            self.pose_graph_edges[(img2_id, img1_id)] = OverlapData(img2_kpts, img1_kpts, np.linalg.inv(H))

            self.img_matches[(img1_id, img2_id)] = mdata 

        
    def optimize(self, central_img_id: int):
        # hardcoded initialization for first camera 
        self.active_node_ids = [0]
        self.pose_graph_nodes[0].f = self.__estimate_camera_initial_focal() 
        self.pose_graph_nodes[0].rot = np.zeros(3)

        print (self.pose_graph_nodes[0].f)

        for i in range(len(self.pose_graph_nodes) - 1):
            # add a new image to optimize
            found_valid = self.__find_next_best_image()
            
            # exist if no new image left to process 
            if not found_valid: break

            # set x0 from current pose estimates 
            x0 = self.__pack_poses() 

            # perform optimization
            res = least_squares(self.__eval_reprojection_error, x0, method='trf', jac='2-point', verbose=2)

            # TODO: check result is valid 
            print(res)

        # unpack final result and store data in pose_graph_nodes
        self.__unpack_poses(res.x)
        
        res = least_squares(self.__eval_reprojection_error, x0, method='trf', f_scale=3, loss='huber', jac='2-point', verbose=2)
        self.__unpack_poses(res.x)

        for i, j, overlap in self.active_edges: 
            res = find_inliers(overlap.H, overlap.src_kpts, overlap.dst_kpts, 3)
            print(f"{i} {j} -> {res[0]}/{len(overlap.src_kpts)}")

        print(self.active_node_ids) 
        for i, pose in enumerate(self.pose_graph_nodes):
            print(f"[{i}]: {pose}")

        # h, w, _ = self.imgs[0].shape
        # T1 = np.array([[1, 0, w/2],
        #                 [0, 1, h/2],
        #                 [0, 0, 1]])
        # invT1 = np.linalg.inv(T1)
        # H = self.pose_graph_nodes[1].get_cam_mat() @ self.pose_graph_nodes[0].get_inv_cam_mat()
        # H = T1 @ H @ invT1
        
        # kpts1 = self.img_matches[(0, 1)].img1_kpts
        # kpts2 = self.img_matches[(0, 1)].img2_kpts
        # print(H)
        # od = self.pose_graph_edges[(0, 1)]
        # print(find_inliers(H, kpts1, kpts2, 3))
        # print(find_inliers(self.img_matches[(0, 1)].H, kpts1, kpts2, 3))

    def __find_next_best_image(self) -> bool: 
        """
            Finds the image_i which shares the highest number of matching features with
            the images currently in active_graph_nodes.

            The camera pose associated with image_i is updated accordingly: 
                - find image_j in active_graph_nodes with highest number of overlapping features
                - copy f from pose_j in active_graph_nodes 
                - compute rot based on pose_j in active graph_nodes 
        """
        
        max_num_overlaps, best_i = 0, -1 
        for i in range(len(self.pose_graph_nodes)):
            # ignore i already active 
            if i in self.active_node_ids: continue

            # compute total number of overlaps
            num_overlaps = 0 
            for j in self.active_node_ids:
                if (j, i) in self.pose_graph_edges:
                    od = self.pose_graph_edges[(j, i)]
                    num_overlaps += len(od.src_kpts)
            # update current best i
            if num_overlaps > max_num_overlaps:
                max_num_overlaps, best_i = num_overlaps, i

        if DEBUG_ENABLED():
            print(f"Found next image {best_i} with {max_num_overlaps} overlapping points")

        # if we don't have a valid best_i early exit
        if best_i == -1: return False

        # store new active node
        self.active_node_ids.append(best_i)

        max_num_overlaps, best_j = 0, -1 
        for j in self.active_node_ids: 
            if (j, best_i) in self.pose_graph_edges: 
                od = self.pose_graph_edges[(j, best_i)]

                # attach new edges to active list
                self.active_edges.append((j, best_i, od))

                if len(od.src_kpts) > max_num_overlaps:
                    max_num_overlaps, best_j = len(od.src_kpts), j 

        if DEBUG_ENABLED():
            print(f"Pose {best_i} inherits from pose {best_j}")

        # Update pose i based on pose j 
        pose_i, pose_j = self.pose_graph_nodes[best_i], self.pose_graph_nodes[best_j] 
        Hij = self.pose_graph_edges[(best_j, best_i)].H

        pose_i.f = pose_j.f
        pose_i.rot = CameraPose.rot_from_homography(pose_j, pose_i, Hij)
        return True

    def __pack_poses(self) -> np.ndarray: 
        dim =  4 * len(self.active_node_ids)
        # vector for packed poses
        x = np.zeros(dim)

        for i in range(len(self.active_node_ids)):
            node_id = self.active_node_ids[i]
            pose = self.pose_graph_nodes[node_id]
            x[4*i] = pose.f
            x[4*i + 1: 4*(i+1)] = pose.rot
        return x

    def __unpack_poses(self, x: np.array): 
        for i in range(len(self.active_node_ids)):
            node_id = self.active_node_ids[i]
            self.pose_graph_nodes[node_id].f = x[4*i] 
            self.pose_graph_nodes[node_id].rot = x[4*i + 1: 4*(i+1)]

    def __eval_reprojection_error(self, x: np.array) -> np.array:
        # update camera poses with current estimate
        self.__unpack_poses(x)

        errors = []
        for ni, nj, overlap in self.active_edges:
            # skip connections with inactive nodes
            if nj not in self.active_node_ids: 
                continue
            
            # store poses & kpts 
            kpts_i, kpts_j = overlap.src_kpts, overlap.dst_kpts
            pose_i = self.pose_graph_nodes[ni]
            pose_j = self.pose_graph_nodes[nj]

            # compute mapping from i to j:
            Mji = pose_j.get_cam_mat() @ pose_i.get_inv_cam_mat()  

            # transform kpts_i int j frame
            kpts_j_est = transform_points(Mji, kpts_i)

            # compute residual
            res = kpts_j - kpts_j_est
            errors.append(np.linalg.norm(res, axis=1))
        return np.hstack(errors)
    
    def __recenter_keypoints(self, img1: np.ndarray, img2: np.ndarray, 
                             kpts1: np.ndarray, kpts2: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w, _ = img2.shape 
        T2 = np.array([[1, 0, w/2], 
                        [0, 1, h/2], 
                        [0, 0 , 1]])
        invT2 = np.linalg.inv(T2)

        h, w, _ = img1.shape
        T1 = np.array([[1, 0, w/2],
                        [0, 1, h/2], 
                        [0, 0, 1]])
        invT1 = np.linalg.inv(T1)

        kpts1_shift = transform_points(invT1, kpts1)
        kpts2_shift = transform_points(invT2, kpts2)
        H_shift = invT2 @ H @ T1
        return kpts1_shift, kpts2_shift, H_shift 


    def __estimate_camera_initial_focal(self):
        f_est = []

        for overlap_data in self.pose_graph_edges.values(): 
            f = self.__estimate_focal_length(overlap_data.H)
            if f is not None:
                f_est.append(f)

        print(f_est)
        if len(f_est) > 0:
            return np.median(f_est) 

        # could not estimate focal length set value to something sane?  
        return -1 

    def __estimate_focal_length(self, M: np.ndarray) -> float|None:
        """
            Estimates focal length of camera 0 based on the mapping matrix M (it is special case of homography matrix)
            which can be decomposed into M = V_1 * R_10 * V_0^-1
            Returns tuple of focal lengths
            Implementation based on: 
                https://imkaywu.github.io/blog/2017/10/focal-from-homography/
                https://stackoverflow.com/questions/71035099/how-can-i-call-opencv-focalsfromhomography-from-python
        """
        assert M.shape == (3, 3)
        m = np.ravel(M) 

        d1 = m[0] * m[3] + m[1] * m[4]
        d2 = m[0] * m[0] + m[1] * m[1] - m[3] * m[3] - m[4] * m[4]

        v1 = -m[2] * m[5] / d1 if d1 != 0.0 else -1.0
        v2 = (m[5] * m[5] - m[2] * m[2]) / d2 if d2 != 0.0 else -1.0
        
        f0 = None 
        if v1 < v2: 
            v1, v2 = v2, v1
            d1, d2 = d2, d1
        if v1 > 0 and v2 > 0: f0 = np.sqrt(v1 if abs(d1) > abs(d2) else v2)
        elif v1 > 0: f0 = np.sqrt(v1) 
        return f0
                
import matplotlib.pyplot as plt

class PoseVisualizer():

    @staticmethod
    def display(poses: list[CameraPose], imgs: list[np.ndarray]):
        ax = plt.figure().add_subplot(projection='3d')

        for i in range(len(poses)):
            pts = PoseVisualizer.__get_camera_helper_pts(poses[i], imgs[i])
            x, y, z = pts.T 
            ax.plot(x, y, z, label='parametric curve')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Pose Visualization')
        ax.set_aspect('equal')
        plt.show()
    
    @staticmethod
    def __get_camera_helper_pts(pose: CameraPose, img: np.ndarray):
        h, w = img.shape[0]/2, img.shape[1]/2
        print(h, w)
        f = pose.f

        # camera gizmo points
        pts = np.array([
            # top
            [0, 0, 0], [-w, h, f], [ w, h, f],
            # right
            [0, 0, 0], [w, h, f], [w,-h, f],
            # bottom
            [0, 0, 0], [w,-h, f], [-w,-h,f],
            # left
            [0, 0, 0], [-w,-h,f], [-w, h, f]
        ]) 
        # return rotated points
        return (so3.exp(pose.rot).T @ pts.T).T 

