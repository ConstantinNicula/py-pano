import numpy as np
from collections import deque
from scipy.optimize import least_squares
from dataclasses import dataclass
from matcher import MatchData

import so3
from homography import *

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

    @classmethod
    def rot_from_homography(pose1, pose2, H21: np.ndarray) -> np.ndarray:
        # find best pose2.rot that best explains H21 (homography from 1 to 2)
        # H21 = K2 * R2 * R1^-1 * K1^-1 
        # R2 * R1^-1 = K2^-1 * H21 * K1
        # R2 = K2^-1 * H21 * K1 * R1  = K2^-1 * H21 * CM1
        R21 = pose2.get_inv_k_mat() @ H21 @ pose1.get_cam_mat()
        return so3.log(R21)

class Bundler: 
    def __init__(self, imgs: list[np.ndarray]|None = None, img_matches: list[MatchData]|None = None):
        # storage for images
        self.imgs = [] 
        
        # camera poses are graph nodes
        self.pose_graph_nodes = []

        # map of which represents pose graph data 
        self.pose_graph_edges: dict[int, list[tuple[int, OverlapData]]]= {} 

        # store provided data
        if imgs: self.set_images(imgs)
        if img_matches: self.set_match_data(img_matches)
        
        # ids nodes which are currently being processed
        self.active_graph_nodes = []

    def set_images(self, imgs: list[np.ndarray]):
        assert imgs is not None
        self.imgs = imgs
        self.pose_graph_nodes = [CameraPose() for _ in imgs]
        for img_id in range(len(imgs)):
            self.pose_graph_edges[img_id] = []

    def set_match_data(self, img_matches: list[MatchData]):
        assert img_matches is not None

        for mdata in img_matches: 
            img1_id, img2_id = mdata.img1_id, mdata.img2_id
            img1_kpts, img2_kpts, H = mdata.img1_kpts, mdata.img2_kpts, mdata.H

            # recenter points and compute new homography 
            img1_kpts, img2_kpts, H = self.__recenter_keypoints(self.imgs[img1_id], self.imgs[img2_id], 
                                                                img1_kpts, img2_kpts, H) 

            # Store adjacency info and measurement data 
            edge1to2 = (img2_id, OverlapData(img1_kpts, img2_kpts, H))
            edge2to1 = (img1_id, OverlapData(img2_kpts, img1_kpts, np.linalg.inv(H)))

            self.pose_graph_edges[img1_id].append(edge1to2)
            self.pose_graph_edges[img2_id].append(edge2to1) 
        
    def optimize(self, central_img_id: int):
        f = self.__estimate_camera_initial_focal()
        print(f)

        # hardcoded initialization for first two cameras
        self.active_graph_nodes = [0, 1, 2, 3]

        self.pose_graph_nodes[0].f = f
        self.pose_graph_nodes[1].f = f
        self.pose_graph_nodes[2].f = f
        self.pose_graph_nodes[3].f = f

        x0 = self.__pack_poses() 

        # perform optimization
        res = least_squares(self.__eval_reprojection_error, x0, method='trf')
        print(res)
        # TODO: check result is valid 
        self.__unpack_poses(res.x)

        print(self.active_graph_nodes) 
        for pose in self.pose_graph_nodes:
            print(pose)

    def __pack_poses(self) -> np.ndarray: 
        dim =  4 * (len(self.active_graph_nodes) - 1) + 1

        # vector for packed poses
        x = np.zeros(dim)

        # active poses contains indices of camera poses, in the order they where included
        node_id = self.active_graph_nodes[0]
        x[0] = self.pose_graph_nodes[node_id].f 

        for i in range(len(self.active_graph_nodes)-1):
            node_id = self.active_graph_nodes[i+1]
            pose = self.pose_graph_nodes[node_id]
            x[4*i + 1] = pose.f
            x[4*i + 2: 4*(i+1) + 1] = pose.rot
        return x

    def __unpack_poses(self, x: np.array): 
        # handle first node separately (no rot data)
        node_id = self.active_graph_nodes[0]
        self.pose_graph_nodes[node_id].f = x[0]

        for i in range(len(self.active_graph_nodes)-1):
            node_id = self.active_graph_nodes[i+1]
            self.pose_graph_nodes[node_id].f = x[4*i + 1] 
            self.pose_graph_nodes[node_id].rot = x[4*i + 2: 4*(i+1) + 1]

    def __eval_reprojection_error(self, x: np.array) -> np.array:
        # update camera poses with current estimate
        self.__unpack_poses(x)

        errors = []
        for ni in self.active_graph_nodes:
            for nj, overlap in self.pose_graph_edges[ni]:

                # skip connections with inactive nodes
                if nj not in self.active_graph_nodes: 
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
        for c_id in self.pose_graph_edges: 
            for _, overlap_data in self.pose_graph_edges[c_id]:
                f = self.__estimate_focal_length(overlap_data.H)
                if f is not None:
                    f_est.append(f)

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
            [0, 0, 0],
            [-w, h, f],
            [ w, h, f],
            # right
            [0, 0, 0],
            [w, h, f],
            [w,-h, f],
            # bottom
            [0, 0, 0],
            [w,-h, f],
            [-w,-h,f],
            # left
            [0, 0, 0],
            [-w,-h,f],
            [-w, h, f]
        ]) 
        # return rotated points
        return (so3.exp(pose.rot).T @ pts.T).T 

