import numpy as np
from homography import *
import so3

def comp_k(f: float) -> np.ndarray:
    return np.array([[f, 0, 0], 
                     [0, f, 0],
                     [0, 0, 1]])

def res(f1: float, phi1: np.ndarray, f2: float, phi2: np.ndarray,  pt1: np.ndarray, pt2: np.ndarray, rav=True) -> np.ndarray:
    T = comp_k(f2) @ so3.exp(phi2) @ so3.exp(phi1).T @ np.linalg.inv(comp_k(f1))
    if rav: 
        return np.ravel((pt2 - transform_points(T, pt1))[:, 0:2])
    return pt2 - transform_points(T, pt1)

def numerical_jacobian(f1: float, phi1: np.ndarray, f2: float, phi2: np.ndarray, pt1: np.ndarray, pt2: np.ndarray)-> np.ndarray:
    h = 1e-6

    J = np.zeros((2, 8))
    # derivatives with respect to focal length
    J[:, 0] = (res(f1+h, phi1, f2, phi2, pt1, pt2) - res(f1-h, phi1, f2, phi2, pt1, pt2)) / (2*h)
    J[:, 4] = (res(f1, phi1, f2+h, phi2, pt1, pt2) - res(f1, phi1, f2-h, phi2, pt1, pt2)) / (2*h)
    
    e1 = np.array([h, 0, 0])
    e2 = np.array([0, h, 0])
    e3 = np.array([0, 0, h])

    # dr/dphi1
    J[:, 1] = (res(f1, phi1 + e1, f2, phi2, pt1, pt2) - res(f1, phi1-e1, f2, phi2, pt1, pt2)) / (2*h)
    J[:, 2] = (res(f1, phi1 + e2, f2, phi2, pt1, pt2) - res(f1, phi1-e2, f2, phi2, pt1, pt2)) / (2*h)
    J[:, 3] = (res(f1, phi1 + e3, f2, phi2, pt1, pt2) - res(f1, phi1-e3, f2, phi2, pt1, pt2)) / (2*h)

    J[:, 5] = (res(f1, phi1, f2, phi2+e1, pt1, pt2) - res(f1, phi1, f2, phi2-e1, pt1, pt2)) / (2*h)
    J[:, 6] = (res(f1, phi1, f2, phi2+e2, pt1, pt2) - res(f1, phi1, f2, phi2-e2, pt1, pt2)) / (2*h)
    J[:, 7] = (res(f1, phi1, f2, phi2+e3, pt1, pt2) - res(f1, phi1, f2, phi2-e3, pt1, pt2)) / (2*h)

    return J

def left_jacobian(phi: np.ndarray) -> np.ndarray:
    theta = np.sqrt(np.dot(phi, phi))
    if np.isclose(theta, 0): return np.eye(3)
    a = phi / theta
    return np.sin(theta) / theta * np.eye(3) + \
           (1 - np.sin(theta) / theta) * np.outer(a, a) + \
           (1 - np.cos(theta)) / theta * so3.hat(a)

    # return np.eye(3) + (1-np.cos(theta))/(theta**2) * so3.hat(phi) + \
    #         (theta - np.sin(theta))/theta**3 * so3.hat(phi) @ so3.hat(phi)

def left_jacobian_approx(phi: np.ndarray) -> np.ndarray:
    return np.eye(3) + 0.5 * so3.hat(phi)


def jac_proj(pt: np.ndarray) -> np.ndarray:
    x, y, z  = pt
    return np.array([
        [1/z, 0, -x/z**2],
        [0, 1/z, -y/z**2]
    ]) 


def real_jac(f1: float, phi1: np.ndarray, f2: float, phi2: np.ndarray, pt1: np.ndarray, pt2: np.ndarray)-> np.ndarray:
    J = np.zeros((2, 8))

    R2 = so3.exp(phi2)
    K2 = comp_k(f2)
    R1 = so3.exp(phi1)
    invK1 = np.linalg.inv(comp_k(f1))

    # constants
    p1_transf = pt1 @ (K2 @ R2 @ R1.T @ invK1).T
    J_proj = jac_proj(p1_transf[0])

    # derivatives with respect to f1
    K1_df1 = np.array([[-1/f1**2, 0, 0], 
                     [0, -1/f1**2, 0], 
                     [0, 0, 0]])
    T_df1 = J_proj @ K2 @ R2 @ R1.T @ K1_df1 @ pt1.T
    J[:, 0] = np.ravel(T_df1)

    # derivatives with respect to f2
    K2_df2 = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
    T_df2 = J_proj @ K2_df2 @ R2 @ R1.T @ invK1 @ pt1.T 
    J[:, 4] = np.ravel(T_df2)

    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    
    JL_R1 = left_jacobian(-phi1)
    print(left_jacobian(phi1))
    print(left_jacobian(-phi1))
    # dR2_dph1
    R1_dphi_1 = -so3.hat(JL_R1 @ e1) @ R1.T 
    T_dphi_1 = J_proj @ K2 @ R2 @ R1_dphi_1 @ invK1 @ pt1.T 
    J[:, 1] = np.ravel(T_dphi_1)

    R1_dphi_2 = -so3.hat(JL_R1 @ e2) @ R1.T 
    T_dphi_2 = J_proj @ K2 @ R2 @ R1_dphi_2 @ invK1 @ pt1.T 
    J[:, 2] = np.ravel(T_dphi_2)

    R1_dphi_3 = -so3.hat(JL_R1 @ e3) @ R1.T 
    T_dphi_3 = J_proj @ K2 @ R2 @ R1_dphi_3 @ invK1 @ pt1.T 
    J[:, 3] = np.ravel(T_dphi_3)

    JL_R2 = left_jacobian(phi2)

    # dR2_dph1
    R2_dphi_1 = so3.hat(JL_R2 @ e1) @ R2 
    T_dphi_1 = J_proj @ K2 @ R2_dphi_1 @ R1.T @ invK1 @ pt1.T 
    J[:, 5] = np.ravel(T_dphi_1)

    R2_dphi_2 = so3.hat(JL_R2 @ e2) @ R2 
    T_dphi_2 = J_proj @ K2 @ R2_dphi_2 @ R1.T @ invK1 @ pt1.T 
    J[:, 6] = np.ravel(T_dphi_2)

    R2_dphi_3 = so3.hat(JL_R2 @ e3) @ R2 
    T_dphi_3 = J_proj @ K2 @ R2_dphi_3 @ R1.T @ invK1 @ pt1.T 
    J[:, 7] = np.ravel(T_dphi_3)


    return -J

def main(): 
    # np.random.seed(1732)
    np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: f"{x:15.3}" ))

    phi1 = np.pi * np.random.randn(3)
    phi2 = np.pi * np.random.randn(3)
    
    f1, f2 = 400, 500

    # err = np.random.randn(3) 
    err = np.zeros(3)
    
    p1 = np.array([[200, 300, 1]])
    p2 = -res(f1, phi1, f2, phi2, p1, np.zeros(3), False) + err 
    
    print(p1, p2)

    reproj_err = res(f1, phi1, f2, phi2, p1, p2)

    J_num = numerical_jacobian(f1, phi1, f2, phi2, p1, p2) 
    J_real = real_jac(f1, phi1, f2, phi2, p1, p2)
    print(J_num)
    print(J_real) 
    print("max error: ", np.max(J_real-J_num))
if __name__ == "__main__":
    main()