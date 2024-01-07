import numpy as np

# Implementation references: 
# https://cvg.cit.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf

def exp(phi: np.ndarray) -> np.ndarray: 
    t = np.sqrt(np.dot(phi, phi))
    ct, st = np.cos(t), np.sin(t)
    n = phi / t if t != 0.0 else np.array([1.0, 0.0, 0.0])
    return ct * np.eye(3) + (1 - ct) * np.outer(n, n)  + st * hat(n) 

# def normalize_mat(R: np.ndarray):
#     # https://math.stackexchange.com/questions/3292034/normalizing-a-quasi-rotation-matrix
#     # https://stackoverflow.com/questions/23080791/eigen-re-orthogonalization-of-rotation-matrix
#     return R * np.linalg.inv(np.sqrt(R.T @ R))


def log(R: np.ndarray) -> np.ndarray:
        cos_angle = np.clip(0.5 * np.trace(R) - 0.5, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        print(f"{angle=:} {R=:}")
        if np.isclose(angle, 0.): 
            return np.zeros(3)

        if np.isclose(angle, np.pi):
            e0 = 1 
            e1 = np.sign(R[0, 1]) 
            e2 = np.sign(R[0, 2])
            w = np.pi * np.array([
                 e0 * np.sqrt(abs(0.5 * (1 + R[0, 0]))),
                 e1 * np.sqrt(abs(0.5 * (1 + R[1, 1]))),
                 e2 * np.sqrt(abs(0.5 * (1 + R[2, 2]))),
            ])
            return w
            # return vee(R - np.identity(3))
        print(np.sin(angle))
        return vee((0.5 * angle / np.sin(angle)) * (R - R.T))

def hat(x: np.ndarray) -> np.ndarray:
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])

def vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]])

def left_jacobian(phi: np.ndarray) -> np.ndarray:
    theta = np.sqrt(np.dot(phi, phi))
    if np.isclose(theta, 0): return np.eye(3)
    a = phi / theta
    return np.sin(theta) / theta * np.eye(3) + \
           (1 - np.sin(theta) / theta) * np.outer(a, a) + \
           (1 - np.cos(theta)) / theta * hat(a)