import numpy as np
from scipy.optimize import least_squares

class Bundler: 
    def __init__(self):
        self.used_img_cnt = 0 
        self.img_data = {}
        self.img_overlap_data = []
        
        
    def add_image(self, img_id:int, img: np.ndarray):
        # no need to store the actual image, store internal id and image shape
        self.img_data[img_id] = [-1, img.shape]

    def add_overlapping_points(self, img_id1: int, img_id2: int, kpts1: np.ndarray, kpts2: np.ndarray):
        assert len(kpts1) == len(kpts2)

        # flag images as used
        if self.img_data[img_id1][0] == -1:
            self.img_data[img_id1][0] = self.__get_next_id()

        if self.img_data[img_id2][0] == -1:
            self.img_data[img_id2][0] = self.__get_next_id()

        # convert to internal ids 
        self.img_overlap_data.append((img_id1, img_id2, kpts1, kpts2))

    def __get_next_id(self): 
        ret = self.used_img_cnt
        self.used_img_cnt += 1
        return ret
    
    def optimize(self):
        #https://imkaywu.github.io/blog/2017/10/focal-from-homography/
        x0 = np.zeros(((self.used_img_cnt-1) * 4 ))
        x0[0:3] = [0.1 +5.23598704e-01, 0.05+5.23598658e-01,  6.63581180e-10]
        x0[3] =500 
        res = least_squares(self.__eval_reprojection_error, x0, method='trf')
        # find unique ids 
        assert len(self.img_overlap_data) > 0 


    def __eval_reprojection_error(self, x: np.array) -> np.array:
        phi_i, focal_i = np.zeros((3, )), x[3]
        phi_j, focal_j = np.zeros((3, )), x[3]

        print(x)
        cam_params = np.reshape(x, (-1, 4))
        errors = []
        for img_i, img_j, kpts_i, kpts_j in self.img_overlap_data:
            # unpack image details
            i, size_i = self.img_data[img_i]
            j, size_j = self.img_data[img_j]

            # unpack camera params
            if i != 0:
                phi_i, focal_i = cam_params[i-1][0:3], cam_params[i-1][3]
            cy_i, cx_i = size_i[0] / 2, size_i[1] / 2

            if j != 0:            
                phi_j, focal_j = cam_params[j-1][0:3], cam_params[j-1][3]
            cy_j, cx_j = size_j[0] / 2, size_j[1] / 2
            
            # compute camera intrinsics and extrinsics
            Ki, Ri = create_cam_mat(focal_i, cx_i, cy_i), create_rot_mat(phi_i)
            Kj, Rj = create_cam_mat(focal_j, cx_j, cy_j), create_rot_mat(phi_j)

            # compute full transform from i to j 
            Tji = Kj @ Rj @ Ri.T @ np.linalg.inv(Ki)
         
            # transform keypoints i into j reference frame 
            kpts_j_est = (Tji @ kpts_i.T).T
            kpts_j_est = (kpts_j_est.T / kpts_j_est[:, 2]).T

            # compute residual
            res = kpts_j - kpts_j_est
            errors.append(np.linalg.norm(res, axis=1))
        print(errors)
        return np.hstack(errors)

def create_cam_mat(f: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[ f,  0, cx], 
                     [ 0,  f, cy],
                     [ 0,  0,  1]] )

def create_rot_mat(phi: np.ndarray) -> np.ndarray: 
    t = np.sqrt(np.dot(phi, phi))
    ct, st = np.cos(t), np.sin(t)
    n = phi / t if t != 0.0 else np.array([1.0, 0.0, 0.0])
    return ct * np.eye(3) + (1 - ct) * np.outer(n, n)  + st * skew(n) 

def skew(x: np.ndarray) -> np.ndarray:
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])


if __name__ == "__main__":
    bundler = Bundler()
    h, w = 1000, 1000
    bundler.add_image(0, np.zeros((h, w)))
    bundler.add_image(1, np.zeros((h, w)))

    f = 700
    K = create_cam_mat(f, w/2, h/2) 
    R = create_rot_mat(np.pi/6 * np.array([1, 1, 0]))
    pts = []
    for i in range(100):
        pts.append([np.random.randint(0, w/2), np.random.randint(0, h/2), 1]) 
    p1 = np.array(pts)

    #p1 = np.array([[100, 100, 1], [200, 200, 1], [300, 300, 1], [400, 400, 1]])
    p2 = (K @ R @ np.eye(3) @ np.linalg.inv(K) @ p1.T).T
    p2 = (p2.T / p2[:, 2]).T
    print(p1, '\n', p2)

    bundler.add_overlapping_points(0, 1, p1, p2 )
    bundler.optimize()
