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
        # camera intrinsics parameters
        self.f = f
        # camera extrinsic parameters
        self.rot = rot 

    def get_cam_mat(self):
        return CameraPose.cam_mat(self.f, self.rot) 

    def get_inv_cam_mat(self): 
       return CameraPose.inv_cam_mat(self.f, self.rot) 

    def get_k_mat(self): 
        return CameraPose.k_mat(self.f) 

    def get_inv_k_mat(self):
        return CameraPose.inv_k_mat(self.f)

    def __repr__(self):
        return f"f: {self.f} rot: {self.rot}"

    @staticmethod
    def cam_mat(f: float, rot: np.ndarray) -> np.ndarray: 
        return  CameraPose.k_mat(f) @ so3.exp(rot)    

    @staticmethod
    def inv_cam_mat(f: float, rot: np.ndarray) -> np.ndarray: 
        return  so3.exp(rot).T @ CameraPose.inv_k_mat(f)    

    @staticmethod
    def k_mat(f: float) -> np.ndarray:
        return np.array([[f, 0, 0], 
                         [0, f, 0], 
                         [0, 0, 1]])
    @staticmethod
    def inv_k_mat(f: float) -> np.ndarray:
        return np.array([[1/f, 0, 0], 
                         [0, 1/f, 0], 
                         [0, 0, 1]])

    @staticmethod
    def rot_from_homography(pose1, pose2, H21: np.ndarray) -> np.ndarray:
        # find best pose2.rot that best explains H21 (homography from 1 to 2)
        # H21 = K2 * R2 * R1^-1 * K1^-1 
        # R2 * R1^-1 = K2^-1 * H21 * K1
        # R21 = K2^-1 * H21 * K1

        R21 = pose2.get_inv_k_mat() @ H21 @ pose1.get_k_mat()
        w21 = so3.log(so3.normalize_mat(R21))
        print(w21)
        return so3.log( so3.exp(w21) @ so3.exp(pose1.rot)) 

