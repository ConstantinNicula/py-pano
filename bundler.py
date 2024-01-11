from scipy.sparse import csr_matrix
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
        # stores mapping between n_i and x_i
        self.node_state_offset: np.ndarray = None

    def set_images(self, imgs: list[np.ndarray]):
        assert imgs is not None
        self.imgs = imgs
        self.pose_graph_nodes = [CameraPose() for _ in imgs]
        self.node_state_offset = np.zeros(len(self.pose_graph_nodes), dtype=np.int32)

    def set_match_data(self, img_matches: list[MatchData]):
        assert img_matches is not None
        self.img_matches = {} 

        for mdata in img_matches: 
            img1_id, img2_id = mdata.img1_id, mdata.img2_id
            img1_kpts, img2_kpts, H = mdata.img1_kpts, mdata.img2_kpts, mdata.H

            # recenter points and compute new homography 
            img1_kpts_rec, img2_kpts_rec, H_rec = self.__recenter_keypoints(self.imgs[img1_id], self.imgs[img2_id], 
                                                                img1_kpts, img2_kpts, H) 

            # compute inverse mapping
            invH_rec = np.linalg.inv(H_rec) 
            invH_rec /= invH_rec[-1, -1]
            
            # Store adjacency info and measurement data 
            self.pose_graph_edges[(img1_id, img2_id)] = OverlapData(img1_kpts_rec, img2_kpts_rec, H_rec)
            self.pose_graph_edges[(img2_id, img1_id)] = OverlapData(img2_kpts_rec, img1_kpts_rec, invH_rec)
            self.img_matches[(img1_id, img2_id)] = mdata 

        
    def optimize(self, central_img_id: int):
        # hardcoded initialization for first camera 
        self.pose_graph_nodes[0].f = self.__estimate_camera_initial_focal()
        self.pose_graph_nodes[0].rot = np.zeros(3)
        self.__mark_active_node(0)

        print(f">>>>>>> {self.pose_graph_nodes[0].f} <<<<<<<<<<<<")
        for i in range(len(self.pose_graph_nodes) - 1):
            # add a new image to optimize
            found_valid = self.__find_next_best_image()

            # exist if no new image left to process 
            if not found_valid: break

            # set x0 from current pose estimates 
            x0 = self.__pack_poses() 

            # perform optimization
            scale = np.full(x0.shape, np.pi/16)
            scale[::4] = np.mean(x0[::4])/10 
            res = least_squares(self.__eval_reprojection_error, x0, 
                                jac = self.__eval_jacobian,  
                                x_scale = scale,
                                method='lm', verbose=2)
            # unpack final result and store data in pose_graph_nodes
            if DEBUG_ENABLED(): print(res)
            self.__unpack_poses(res.x)

        # Do a final refine step
        x0 = self.__pack_poses() 
        res = least_squares(self.__eval_reprojection_error, x0, 
                            jac = self.__eval_jacobian, 
                            method='trf', x_scale ='jac', f_scale=3, 
                            loss='huber', verbose=2)
        self.__unpack_poses(res.x)

        if DEBUG_ENABLED():
            print(res)

        # Remove unnecessary rotations
        self.__remove_camera_rotations()

        for i, j, overlap in self.active_edges: 
            final_H = self.pose_graph_nodes[j].get_cam_mat() @ self.pose_graph_nodes[i].get_inv_cam_mat()
            r = overlap.dst_kpts - transform_points(final_H, overlap.src_kpts)
            print(np.linalg.norm(r))

        print(self.active_node_ids) 
        for i, pose in enumerate(self.pose_graph_nodes):
            print(f"[{i}]: {pose}")

    def __mark_active_node(self, id: int): 
        self.active_node_ids.append(id)
        self.node_state_offset[id] = len(self.active_node_ids)-1    
    
    def __find_next_best_image(self) -> bool: 
        """
            Finds the image_i which shares the highest number of matching features with
            the images currently in active_graph_nodes.

            The camera pose associated with image_i is updated accordingly: 
                - find image_j in active_graph_nodes with highest number of overlapping features
                - copy f from pose_j in active_graph_nodes 
                - compute rot based on pose_j in active graph_nodes 
        """

        # 1) Find the index i of the next candidate image
        max_matching_pts, best_i = 0, -1 
        for i in range(len(self.pose_graph_nodes)):
            # ignore i already active 
            if i in self.active_node_ids: continue

            # compute total number of overlaps
            matching_pts = 0 
            for j in self.active_node_ids:
                if (j, i) in self.pose_graph_edges:
                    od = self.pose_graph_edges[(j, i)]
                    matching_pts += len(od.src_kpts)
            # update current best i
            if matching_pts > max_matching_pts:
                max_matching_pts, best_i = matching_pts, i

        if DEBUG_ENABLED(): print(f"Found next image {best_i} with {max_matching_pts} overlapping points")

        # If we don't have a valid best_i early exit
        if best_i == -1: return False

        # 2) Find the index j of the best matching ref image 
        relative_rots = []
        max_matching_pts, best_j = 0, -1 
        for j in self.active_node_ids: 
            if (j, best_i) in self.pose_graph_edges: 
                # store poses of overlapping images
                pose_i = self.pose_graph_nodes[best_i]
                pose_j = self.pose_graph_nodes[j]
                Hji = self.pose_graph_edges[(j, best_i)].H
                relative_rots.append(CameraPose.rot_from_homography(pose_j, pose_i, Hji))
                
                # extract overlap data
                od = self.pose_graph_edges[(j, best_i)]

                # attach new edges to active list
                self.active_edges.append((j, best_i, od))

                if len(od.src_kpts) > max_matching_pts:
                    max_matching_pts, best_j = len(od.src_kpts), j 
        # 3) Compute pose of camera i 
        # Flag node as active
        self.__mark_active_node(best_i)

        # Update pose i based on pose j 
        pose_i = self.pose_graph_nodes[best_i] 
        pose_j = self.pose_graph_nodes[best_j]
        pose_i.f = pose_j.f

        # Compute ideal rotation
        if len(relative_rots) == 1: # inherit rotation from pose_j
            pose_i.rot = relative_rots[0]
            if DEBUG_ENABLED(): print(f"Pose {best_i} inherits from pose {best_j}")
        else: 
            pose_i.rot = self.__compute_average_rotation(relative_rots)
            if DEBUG_ENABLED(): print(f"Pose {best_i} computed from average pose")
        return True

    def __compute_average_rotation(self, relative_rots: list[np.ndarray]) -> np.ndarray:
        x_axis_avg = np.zeros(3)
        z_axis_avg = np.zeros(3)

        # loop over poses and accumulate average axis directions
        for rot in relative_rots:
            # R is local to world matrix
            R = so3.exp(rot)

            # columns of R.T are axis vectors for pose_i
            x_axis_avg += R[0]
            z_axis_avg += R[2] 

        # average out axis vectors
        x_axis_avg /= np.linalg.norm(x_axis_avg) 
        z_axis_avg /= np.linalg.norm(z_axis_avg) 

        # compute 'average' rotation matrix
        # cross(x, y) = z
        y_axis_avg = -np.cross(x_axis_avg, z_axis_avg)        
        z_axis_avg = np.cross(x_axis_avg, y_axis_avg)

        # Compute final rotation matrix
        R_avg = np.array([x_axis_avg, 
                          y_axis_avg, 
                          z_axis_avg])
        return so3.log(R_avg).T


    def __remove_camera_rotations(self):
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

        # Compute up vector as the null vector of 
        # the covariance matrix of camera X vectors
        X = np.zeros((3, 3))
        for ni in self.active_node_ids:
            # Rotation from global RF to camera ni RF
            Ri = so3.exp(self.pose_graph_nodes[ni].rot)

            # X axis is column 0 of Ri.T or row 0 of Ri
            X += np.outer(Ri[0], Ri[0]) 

        # Find null space of X and extract up vector u
        _, _, Vh = np.linalg.svd(X)  
        u = Vh[-1, :]

        # make sure u is pointing in the direction of camera y axis 
        ref_i = self.active_node_ids[0]
        ref_y = so3.exp(self.pose_graph_nodes[ref_i].rot)[1]
        if np.dot(u, ref_y) < 0:
            u *= -1

        # compute alignment rotations 
        w = np.cross(u, np.array([0, 1, 0]))
        n = w / np.linalg.norm(w)
        c = np.dot(u, np.array([0, 1, 0])) 
        s = np.linalg.norm(w)

        # aligns u to camera y axis (global to local ref) 
        R_align = c * np.eye(3) + (1 - c) * np.outer(n, n) + s * so3.hat(n) 

        # apply rotation to all nodes
        for ni in self.active_node_ids:
            pose_i = self.pose_graph_nodes[ni]
            pose_i.rot = so3.log(so3.exp(pose_i.rot) @ R_align.T)

    def __pack_poses(self) -> np.ndarray: 
        dim =  4 * len(self.active_node_ids)
        # vector for packed poses
        x = np.zeros(dim)

        for i, node_id in enumerate(self.active_node_ids):
            pose = self.pose_graph_nodes[node_id]
            x[4*i] = pose.f
            x[4*i + 1: 4*(i+1)] = pose.rot
        return x

    def __unpack_poses(self, x: np.array): 
        for i, node_id in enumerate(self.active_node_ids):
            self.pose_graph_nodes[node_id].f = x[4*i] 
            self.pose_graph_nodes[node_id].rot = x[4*i + 1: 4*(i+1)]

    def __eval_reprojection_error(self, x: np.array) -> np.array:
        # update camera poses with current estimate
        x = np.reshape(x, (-1, 4))

        # preallocate error array
        N = np.sum([len(od.src_kpts) for _, _, od in self.active_edges])
        err, err_offset = np.zeros(2 * N), 0 
        for ni, nj, overlap in self.active_edges:
            # get the true state offset
            oi = self.node_state_offset[ni]
            oj = self.node_state_offset[nj]

            # unpack pose data
            f_i, rot_i = x[oi, 0], x[oi, 1:4]  
            f_j, rot_j = x[oj, 0], x[oj, 1:4]

            # get kpts 
            kpts_i, kpts_j = overlap.src_kpts, overlap.dst_kpts

            # compute mapping from i to j:
            Mji = CameraPose.cam_mat(f_j, rot_j) @ CameraPose.inv_cam_mat(f_i, rot_i)

            # transform kpts_i int j frame
            kpts_j_est = transform_points(Mji, kpts_i)

            # compute residual
            res = kpts_j - kpts_j_est
            
            # store result 
            nr_err_terms = 2 * len(overlap.src_kpts)
            err[err_offset: err_offset + nr_err_terms] = np.ravel(res[:, 0:2])
            err_offset += nr_err_terms 
        return err 


    def __eval_jacobian(self, x: np.ndarray) -> np.ndarray:
        # update camera poses with the current state 
        x = np.reshape(x, (-1, 4))

        # preallocate Jacobian matrix
        N = np.sum([len(od.src_kpts) for _, _, od in self.active_edges])
        M = 4 * len(x)

        J, ro = np.zeros((2 * N, M)), 0 
        for ni, nj, overlap in self.active_edges:
            # get state offset 
            oi = self.node_state_offset[ni]
            oj = self.node_state_offset[nj]

            # unpack pose data
            f_i, rot_i = x[oi, 0], x[oi, 1:4]  
            f_j, rot_j = x[oj, 0], x[oj, 1:4]

            # get kpts 
            kpts_i = overlap.src_kpts

            # Compute J blocks 
            Jb_i, Jb_j = self.__compute_jacobian_blocks(f_i, rot_i, f_j, rot_j, kpts_i)

            # Store blocks
            nr_terms = 2 * len(kpts_i) 
            J[ro:ro + nr_terms, 4*oi: 4*(oi+1)] = Jb_i 
            J[ro:ro + nr_terms, 4*oj: 4*(oj+1)] = Jb_j 
            ro += nr_terms 
        return J

    def __compute_jacobian_blocks(self, f_i: float, rot_i: np.ndarray, 
                                        f_j: float, rot_j: np.ndarray, 
                                        kpts_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  
        # Refs for rotation partial derivatives:
        # https://natanaso.github.io/ece276a2020/ref/ECE276A_12_SO3_SE3.pdf
        
        N = len(kpts_i)
        J_i, J_j = np.zeros((2*N, 4)), np.zeros((2*N, 4)) 

        invRi = so3.exp(rot_i).T
        invKi = CameraPose.inv_k_mat(f_i) 
        Rj = so3.exp(rot_j)
        Kj = CameraPose.k_mat(f_j)
        
        # Common values 
        kpts_i_stack = np.reshape(kpts_i, (N, 3, 1))
        kpts_i_transf = kpts_i @ (Kj @ Rj @ invRi @ invKi).T
        J_proj = self.__compute_projection_jacobian(kpts_i_transf)
        e = np.eye(3)

        # Compute Jacobian block for pose i 
        # Derivatives with respect to f1
        dInvKi = np.array([[-1/f_i**2, 0, 0], 
                           [0, -1/f_i**2, 0], 
                           [0, 0, 0]])
        J_i[:, 0] = np.ravel(J_proj @ Kj @ Rj @ invRi @ dInvKi @ kpts_i_stack)

        # Derivative with respect to element of pose_1.rot 
        JL_Ri = so3.left_jacobian(-rot_i)
        for i in range(3):
            dInvRi = -so3.hat(JL_Ri @ e[i]) @ invRi 
            J_i[:, i + 1] = np.ravel(J_proj @ Kj @ Rj @ dInvRi @ invKi @ kpts_i_stack)

        # Compute jacobian block for pose j
        # derivatives with respect to f2
        dKj_df = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]])
        J_j[:, 0] = np.ravel(J_proj @ dKj_df @ Rj @ invRi @ invKi @ kpts_i_stack)

        JL_Rj = so3.left_jacobian(rot_j)
        # compute derivative of R2 with respect to phi_i
        for i in range(3):
            dRj = so3.hat(JL_Rj @ e[i]) @ Rj 
            J_j[:, i + 1] = np.ravel(J_proj @ Kj @ dRj @ invRi @ invKi @ kpts_i_stack)

        return -J_i, -J_j

    def __compute_projection_jacobian(self, pts: np.ndarray) -> np.ndarray:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        N = len(pts)
        J = np.zeros((2*N, 3)) 
        # 1/z, 0, -x/z**2
        J[::2, 0] = 1 / z 
        J[::2, 2] = -x/np.power(z, 2)
        # 0, 1/z, -y/z**2
        J[1::2,1] = 1 / z 
        J[1::2,2] = -y/np.power(z, 2) 
        return np.reshape(J, (N, 2, 3))

    def __recenter_keypoints(self, img1: np.ndarray, img2: np.ndarray, 
                             kpts1: np.ndarray, kpts2: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = img2.shape[0], img2.shape[1]
        T2 = np.array([[1, 0, w/2], 
                        [0, 1, h/2], 
                        [0, 0 , 1]])
        invT2 = np.linalg.inv(T2)

        h, w = img1.shape[0], img1.shape[1]
        T1 = np.array([[1, 0, w/2],
                        [0, 1, h/2], 
                        [0, 0, 1]])
        invT1 = np.linalg.inv(T1)

        kpts1_shift = transform_points(invT1, kpts1)
        kpts2_shift = transform_points(invT2, kpts2)
        H_shift = invT2 @ H @ T1
        H_shift /= H_shift[-1, -1] 
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
    def display(poses: list[CameraPose], imgs: list[np.ndarray], overlap_pts: list ):
        ax = plt.figure().add_subplot(projection='3d')

        for i in range(len(poses)):
            pts = PoseVisualizer.__get_camera_helper_pts(poses[i], imgs[i])
            x, y, z = pts.T 
            ax.plot(x, y, z)

        for i, j, od in overlap_pts:
            print(f"{i=:}, {j=:}")
            ptsi = od.src_kpts.copy()
            ptsi[:, 2] =poses[i].f 

            ptsi_rot = ptsi @ (so3.exp(poses[i].rot).T).T
            x, y, z = ptsi_rot.T
            ax.scatter3D(x, y, z)

            ptsj = transform_points(poses[j].get_cam_mat() @ poses[i].get_inv_cam_mat(), od.src_kpts)
            ptsj[:, 2] = poses[j].f

            ptsj_rot = ptsj @ (so3.exp(poses[j].rot).T).T
            x, y, z = ptsj_rot.T
            ax.scatter3D(x, y, z)

        # # render up vector: 
        # x, y, z = np.array([[0, 0, 0], up * 2000]).T 
        # ax.plot(x, y, z)

        # # render x vector 
        # vecs = np.zeros((2*len(poses), 3)) 
        # for i in range(len(poses)): 
        #     vecs[2*i, :] = 2000 *so3.exp(poses[i].rot)[0]

        # x, y, z = vecs.T 
        # ax.plot(x, y, z)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Pose Visualization')
        ax.set_aspect('equal')
        plt.show()
    
    @staticmethod
    def __get_camera_helper_pts(pose: CameraPose, img: np.ndarray):
        h, w = img.shape[0]/2, img.shape[1]/2
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
            [0, 0, 0], [-w,-h,f], [-w, h, f],
            # arrow
            [-w, -h, f], [-w/2, -h, f], [0, -h-w/3, f], [w/3, -h, f]
        ]) 
        # return rotated points
        return (so3.exp(pose.rot).T @ pts.T).T 

