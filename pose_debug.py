                
import matplotlib.pyplot as plt
import numpy as np

from pose import CameraPose
from homography import *
import so3

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

        # Camera gizmo points
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
        # Return rotated points
        return (so3.exp(pose.rot).T @ pts.T).T 

