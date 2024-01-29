from scipy.sparse import csr_matrix
from scipy.optimize import least_squares
from dataclasses import dataclass

import so3
from matcher import MatchData
from pose import CameraPose
from homography import *
from debug_utils import *
from img_uitls import *

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

class Bundler: 
    def __init__(self, imgs: list[np.ndarray]|None = None, img_matches: list[MatchData]|None = None):
        # Storage for images
        self.imgs = [] 
        
        # Camera poses are graph nodes
        self.pose_graph_nodes: list[CameraPose] = []

        # Adjacency (overlap) data 
        self.pose_graph_edges: dict[tuple[int, int], OverlapData]= {} 

        # Store provided data
        if imgs: self.set_images(imgs)
        if img_matches: self.set_match_data(img_matches)
        
        # Ids nodes which are currently being processed
        self.active_node_ids: list[int] = []
        self.active_edges: list[tuple(int, int, OverlapData)] = [] 

        # Stores mapping between n_i and x_i
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

            # Recenter points and compute new homography 
            img1_kpts_rec, img2_kpts_rec, H_rec = self.__recenter_keypoints(self.imgs[img1_id], self.imgs[img2_id], 
                                                                img1_kpts, img2_kpts, H) 

            # Compute inverse mapping
            invH_rec = np.linalg.inv(H_rec) 
            invH_rec /= invH_rec[-1, -1]
            
            # Store adjacency info and measurement data 
            self.pose_graph_edges[(img1_id, img2_id)] = OverlapData(img1_kpts_rec, img2_kpts_rec, H_rec)
            self.pose_graph_edges[(img2_id, img1_id)] = OverlapData(img2_kpts_rec, img1_kpts_rec, invH_rec)
            self.img_matches[(img1_id, img2_id)] = mdata 

        
    def optimize(self, st_image_id: int = 0) -> list[CameraPose]:
        # Cleanup internal vars
        self.active_node_ids = []
        self.active_edges = []
        self.node_state_offset = np.zeros_like(self.node_state_offset)

        # Initialize first camera pose 
        self.pose_graph_nodes[st_image_id].f = self.__estimate_camera_initial_focal()
        self.pose_graph_nodes[st_image_id].rot = np.zeros(3)
        self.__mark_active_node(st_image_id)

        print(f">>>>>>> {self.pose_graph_nodes[0].f} <<<<<<<<<<<<")
        for i in range(len(self.pose_graph_nodes) - 1):
            # Add a new image to optimize
            found_valid = self.__find_next_best_image()

            # Exist if no new image left to process 
            if not found_valid: break

            # Set x0 from current pose estimates 
            x0 = self.__pack_poses() 

            # Perform optimization
            # scale = np.full(x0.shape, np.pi/16)
            # scale[::4] = np.mean(x0[::4])/10 
            res = least_squares(self.__eval_reprojection_error, x0, 
                                jac = self.__eval_jacobian,  
                                # x_scale = scale,
                                x_scale ='jac',
                                method='lm', verbose=2)
            # Unpack result and store data in pose_graph_nodes
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

        return self.pose_graph_nodes

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
            # Ignore i already active 
            if i in self.active_node_ids: continue

            # Compute total number of overlaps
            matching_pts = 0 
            for j in self.active_node_ids:
                if (j, i) in self.pose_graph_edges:
                    od = self.pose_graph_edges[(j, i)]
                    matching_pts += len(od.src_kpts)
            # Update current best i
            if matching_pts > max_matching_pts:
                max_matching_pts, best_i = matching_pts, i

        if DEBUG_ENABLED(): print(f"Found next image {best_i} with {max_matching_pts} overlapping points")

        # If we don't have a valid best_i early exit
        if best_i == -1: return False

        # 2) Find the index j of the best matching ref image 
        estimated_rots = []
        max_matching_pts, best_j = 0, -1 
        for j in self.active_node_ids: 
            if (j, best_i) in self.pose_graph_edges: 
                # Store poses of overlapping images
                pose_i = self.pose_graph_nodes[best_i]
                pose_j = self.pose_graph_nodes[j]
                Hji = self.pose_graph_edges[(j, best_i)].H
                estimated_rots.append(CameraPose.rot_from_homography(pose_j, pose_i, Hji))
                
                # Extract overlap data
                od = self.pose_graph_edges[(j, best_i)]

                # Attach new edges to active list
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
        if len(estimated_rots) == 1: # inherit rotation from pose_j
            pose_i.rot = estimated_rots[0]
            if DEBUG_ENABLED(): print(f"Pose {best_i} inherits from pose {best_j}")
        else: 
            pose_i.rot = self.__compute_average_rotation(estimated_rots)
            if DEBUG_ENABLED(): print(f"Pose {best_i} computed from average pose")
            self.__adjust_poses(best_i)
        return True

    def __adjust_poses(self, st_id: int, angle_threshold:float = np.pi/5): 
        """
            Attempts to solve convergence issues by adjusting poses that have an 
            angular deviation higher than a given threshold. Only direct neighbors of 
            st_id pose are adjusted. 
        """
        # 0) extract neighboring pose ids
        st_nbs = self.__get_neighbors_of_pose(st_id) 

        # 1) loop over neighboring poses and adjust
        for i in st_nbs: 
            adjust_needed = False
            i_nbs_rot = []
            
            pose_i = self.pose_graph_nodes[i]
            i_nbs = self.__get_neighbors_of_pose(i)
            for j in i_nbs:
                pose_j = self.pose_graph_nodes[j] 
                
                # Compute relative angle between poses
                rel_rot = so3.log(so3.exp(pose_i.rot) @ so3.exp(pose_j.rot).T) 
                angle = np.fmod(np.linalg.norm(rel_rot), np.pi)

                # If angle exceeds threshold, set update flag 
                if np.abs(angle) > angle_threshold:
                    adjust_needed = True
                
                # Store global rotation of pose j
                i_nbs_rot.append(pose_j.rot)
            
            # Set rotation to average of neighbors
            if adjust_needed:
                print(f"Adjusted pose {i}")
                pose_i.rot = self.__compute_average_rotation(i_nbs_rot)
    
    def __get_neighbors_of_pose(self, i: int):
        return [j for j in self.active_node_ids if (j, i) in self.pose_graph_edges]

    def __compute_average_rotation(self, relative_rots: list[np.ndarray]) -> np.ndarray:
        x_axis_avg = np.zeros(3)
        z_axis_avg = np.zeros(3)

        # Loop over poses and accumulate average axis directions
        for rot in relative_rots:
            # R is local to world matrix
            R = so3.exp(rot)

            # Columns of R.T are axis vectors for pose_i
            x_axis_avg += R[0]
            z_axis_avg += R[2] 

        # Average out axis vectors
        x_axis_avg /= np.linalg.norm(x_axis_avg) 
        z_axis_avg /= np.linalg.norm(z_axis_avg) 

        # Compute 'average' rotation matrix
        # Cross(x, y) = z
        y_axis_avg = -np.cross(x_axis_avg, z_axis_avg)        
        z_axis_avg = np.cross(x_axis_avg, y_axis_avg)

        # Compute final rotation matrix
        R_avg = np.array([x_axis_avg, 
                          y_axis_avg, 
                          z_axis_avg])
        return so3.log(R_avg).T


    def __remove_camera_rotations(self):
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

        # 1) Align horizon 
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

        # Make sure u is pointing in the direction of camera y axis 
        n_id = self.active_node_ids[0]
        ref_y = so3.exp(self.pose_graph_nodes[n_id].rot)[1]
        if np.dot(u, ref_y) < 0: u *= -1

        # Compute alignment rotations 
        w = np.cross(u, np.array([0, 1, 0]))
        n = w / np.linalg.norm(w)
        c = np.dot(u, np.array([0, 1, 0])) 
        s = np.linalg.norm(w)

        # Aligns u to camera y axis (global to local ref) 
        R_align = c * np.eye(3) + (1 - c) * np.outer(n, n) + s * so3.hat(n) 

        # Apply rotation to all nodes
        for ni in self.active_node_ids:
            pose_i = self.pose_graph_nodes[ni]
            pose_i.rot = so3.log(so3.exp(pose_i.rot) @ R_align.T)

        # 2) Remove unnecessary Y rotations
        avg_angle = 0
        for ni in self.active_node_ids:
            # Extract z axis (last row of rotation matrix)
            z_axis = so3.exp(self.pose_graph_nodes[ni].rot)[2]
            # Compute angle of z_axis projection on global Z, X axes 
            ang = np.arctan2(z_axis[2], z_axis[0]) - np.pi/2
            avg_angle += ang

        avg_angle /= len(self.active_node_ids)
        R_align = so3.exp(avg_angle * np.array([0, 1, 0])) 

        # Apply rotations to all nodes
        for ni in self.active_node_ids: 
            pose_i = self.pose_graph_nodes[ni]
            pose_i.rot = so3.log(so3.exp(pose_i.rot) @ R_align.T) 


    def __pack_poses(self) -> np.ndarray: 
        dim =  4 * len(self.active_node_ids)
        # Vector for packed poses
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
        # Update camera poses with current estimate
        x = np.reshape(x, (-1, 4))

        # Preallocate error array
        N = np.sum([len(od.src_kpts) for _, _, od in self.active_edges])
        err, err_offset = np.zeros(2 * N), 0 
        for ni, nj, overlap in self.active_edges:
            # Get the true state offset
            oi = self.node_state_offset[ni]
            oj = self.node_state_offset[nj]

            # Unpack pose data
            f_i, rot_i = x[oi, 0], x[oi, 1:4]  
            f_j, rot_j = x[oj, 0], x[oj, 1:4]

            # Get kpts 
            kpts_i, kpts_j = overlap.src_kpts, overlap.dst_kpts

            # Compute mapping from i to j:
            Mji = CameraPose.cam_mat(f_j, rot_j) @ CameraPose.inv_cam_mat(f_i, rot_i)

            # Transform kpts_i int j frame
            kpts_j_est = transform_points(Mji, kpts_i)

            # Compute residual
            res = kpts_j - kpts_j_est
            
            # Store result 
            nr_err_terms = 2 * len(overlap.src_kpts)
            err[err_offset: err_offset + nr_err_terms] = np.ravel(res[:, 0:2])
            err_offset += nr_err_terms 
        return err 


    def __eval_jacobian(self, x: np.ndarray) -> np.ndarray:
        # Update camera poses with the current state 
        x = np.reshape(x, (-1, 4))

        # Preallocate Jacobian matrix
        N = np.sum([len(od.src_kpts) for _, _, od in self.active_edges])
        M = 4 * len(x)

        J, ro = np.zeros((2 * N, M)), 0 
        for ni, nj, overlap in self.active_edges:
            # Get state offset 
            oi = self.node_state_offset[ni]
            oj = self.node_state_offset[nj]

            # Unpack pose data
            f_i, rot_i = x[oi, 0], x[oi, 1:4]  
            f_j, rot_j = x[oj, 0], x[oj, 1:4]

            # Get kpts 
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
        # Derivatives with respect to f2
        dKj_df = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]])
        J_j[:, 0] = np.ravel(J_proj @ dKj_df @ Rj @ invRi @ invKi @ kpts_i_stack)

        JL_Rj = so3.left_jacobian(rot_j)
        # Compute derivative of R2 with respect to phi_i
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
        # Compute transform for img1 points
        img1_c = image_center(img1)
        T1 = CameraPose.k_mat(1, img1_c)
        invT1 = CameraPose.inv_k_mat(1, img1_c)

        # Compute transforms for img2 points
        img2_c = image_center(img2)
        T2 = CameraPose.k_mat(1, img2_c)
        invT2 = CameraPose.inv_k_mat(1, img2_c)
        
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

        # Could not estimate focal length set value to something sane?  
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
                
