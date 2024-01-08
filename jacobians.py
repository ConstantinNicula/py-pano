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
    h = 1e-4
    N = len(pt1)
    J = np.zeros((2 * N, 8))
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


def proj_deriv(pt: np.ndarray) -> np.ndarray:
    print(pt)
    N = len(pt)
    J = np.zeros((2*N, 3)) 
    x, y, z = pt[:, 0], pt[:, 1], pt[:, 2]
    # 1/z, 0, -x/z**2
    J[::2, 0] = 1 / z 
    J[::2, 2] = -x/np.power(z, 2)
    # 0, 1/z, -y/z**2
    J[1::2,1] = 1 / z 
    J[1::2,2] = -y/np.power(z, 2) 

    return np.reshape(J, (N, 2, 3))

# you can transform points with a stack of rotation matrices 
# pt_transf = pt @ np.transpose(R, (0, 2, 1))



def real_jac(f1: float, phi1: np.ndarray, f2: float, phi2: np.ndarray, pt1: np.ndarray, pt2: np.ndarray)-> np.ndarray:
    N = len(pt1)
    J = np.zeros((2*N, 8))

    R2 = so3.exp(phi2)
    K2 = comp_k(f2)
    R1 = so3.exp(phi1)
    invK1 = np.linalg.inv(comp_k(f1))

    # constants
    p1_transf = pt1 @ (K2 @ R2 @ R1.T @ invK1).T

    J_proj = proj_deriv(p1_transf)

    # convert points to a stack of N column vectors 
    pt1_stack = np.reshape(pt1, (N, 3, 1))

    # derivatives with respect to f1
    K1_df1 = np.array([[-1/f1**2, 0, 0], 
                     [0, -1/f1**2, 0], 
                     [0, 0, 0]])
    T_df1 = J_proj @ K2 @ R2 @ R1.T @ K1_df1 @ pt1_stack 
    J[:, 0] = np.ravel(T_df1)

    # compute derivative of R1 with respect to phi_i 
    e = np.eye(3) 
    JL_R1 = left_jacobian(-phi1)
    for i in range(3):
        dR1_dxi = -so3.hat(JL_R1 @ e[i]) @ R1.T 
        dT_dxi = J_proj @ K2 @ R2 @ dR1_dxi @ invK1 @ pt1_stack 
        J[:, i + 1] = np.ravel(dT_dxi)

    # derivatives with respect to f2
    K2_df2 = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
    T_df2 = J_proj @ K2_df2 @ R2 @ R1.T @ invK1 @ pt1_stack 
    J[:, 4] = np.ravel(T_df2)

    JL_R2 = left_jacobian(phi2)
    # compute derivative of R2 with respect to phi_i
    for i in range(3):
        dR2_dxi = so3.hat(JL_R2 @ e[i]) @ R2 
        dT_dxi = J_proj @ K2 @ dR2_dxi @ R1.T @ invK1 @ pt1_stack 
        J[:, i + 5] = np.ravel(dT_dxi)

    return -J

def main(): 
    # np.random.seed(123)
    np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: f"{x:15.3}" ))

    phi1 = np.pi/6 * np.random.randn(3)
    phi2 = np.pi/6 * np.random.randn(3)
    
    f1, f2 = 400, 500

    err = np.random.randn(3) 
    #err = np.zeros(3)
    
    p1 = np.array([[200, 300, 1], [220, 500, 1]])
    p2 = -res(f1, phi1, f2, phi2, p1, np.zeros(3), False) + err 
    print(p1, p2)

    J_num = numerical_jacobian(f1, phi1, f2, phi2, p1, p2) 
    print(J_num)
    J_real = real_jac(f1, phi1, f2, phi2, p1, p2)
    print(J_real) 
    print("max error: ", np.max(J_real-J_num))
if __name__ == "__main__":
    main()