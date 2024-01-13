import numpy as np
import so3

class CameraPose: 
    """
        Class for keeping track of camera params: 
            - Source image
            - Intrinsic: camera focal length estimate 
            - Extrinsic: Camera rotation estimate 
    """
    def __init__(self, f: float=1.0, rot: np.ndarray = np.zeros(3)):
        # Camera intrinsics parameters
        self.center = np.zeros(2) # optical center (in camera coords) 
        self.f = f

        # Camera extrinsic parameters
        self.rot = rot 

    def set_camera_center(self, center: np.ndarray):
        self.center = center

    def get_cam_mat(self):
        return CameraPose.cam_mat(self.f, self.rot, self.center) 

    def get_inv_cam_mat(self): 
       return CameraPose.inv_cam_mat(self.f, self.rot, self.center) 

    def get_k_mat(self): 
        return CameraPose.k_mat(self.f, self.center) 

    def get_inv_k_mat(self):
        return CameraPose.inv_k_mat(self.f, self.center)
    
    def get_r_mat(self):
        return so3.exp(self.rot)

    def get_inv_r_mat(self):
        return so3.exp(self.rot).T

    def __repr__(self):
        return f"f: {self.f} rot: {self.rot}"

    @staticmethod
    def cam_mat(f: float, rot: np.ndarray, c: np.ndarray = np.zeros(2)) -> np.ndarray: 
        return  CameraPose.k_mat(f, c) @ so3.exp(rot)    

    @staticmethod
    def inv_cam_mat(f: float, rot: np.ndarray, c: np.ndarray = np.zeros(2)) -> np.ndarray: 
        return  so3.exp(rot).T @ CameraPose.inv_k_mat(f, c)    

    @staticmethod
    def k_mat(f: float, c: np.ndarray = np.zeros(2)) -> np.ndarray:
        return np.array([[f, 0, c[0]], 
                         [0, f, c[1]], 
                         [0, 0, 1]])
    @staticmethod
    def inv_k_mat(f: float, c: np.ndarray = np.zeros(2)) -> np.ndarray:
        return np.array([[1/f, 0, -c[0]/f], 
                         [0, 1/f, -c[1]/f], 
                         [0,   0,      1]])

    @staticmethod
    def rot_from_homography(pose1, pose2, H21: np.ndarray) -> np.ndarray:
        # Find best pose2.rot that best explains H21 (homography from 1 to 2)
        # H21 = K2 * R2 * R1^-1 * K1^-1 
        # R2 * R1^-1 = K2^-1 * H21 * K1
        # R21 = K2^-1 * H21 * K1

        R21 = pose2.get_inv_k_mat() @ H21 @ pose1.get_k_mat()
        w21 = so3.log(so3.normalize_mat(R21))
        return so3.log( so3.exp(w21) @ so3.exp(pose1.rot)) 

