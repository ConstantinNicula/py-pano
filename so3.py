import numpy as np

# Implementation references: 
# https://cvg.cit.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf

def exp(phi: np.ndarray) -> np.ndarray: 
    t = np.linalg.norm(phi)
    ct, st = np.cos(t), np.sin(t)
    n = phi / t if t != 0.0 else np.array([0.0, 0.0, 0.0])
    return ct * np.eye(3) + (1 - ct) * np.outer(n, n)  + st * hat(n) 

def normalize_mat(R: np.ndarray):
    # https://math.stackexchange.com/questions/3292034/normalizing-a-quasi-rotation-matrix
    # https://stackoverflow.com/questions/23080791/eigen-re-orthogonalization-of-rotation-matrix
    U, _, V = np.linalg.svd(R)
    return U @ V

def log(R: np.ndarray) -> np.ndarray:
        cos_angle = np.clip(0.5 * np.trace(R) - 0.5, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Use Taylor approximation
        if np.isclose(angle, 0.): 
            return 0.5 * (1 + 1/6 * angle**2 + 7/360 * angle**4) * vee(R - R.T) 

        if np.isclose(angle, np.pi):
            return _log_around_pi(R)
        
        return 0.5 * angle / np.sin(angle) * vee((R - R.T))

def _log_around_pi(R: np.ndarray) -> np.ndarray:
    trR = np.trace(R)
    S = R + R.T + (1 - trR) * np.eye(3)
    n0 = np.sqrt(abs(S[0, 0] / (3 - trR)))
    n1 = np.sqrt(abs(S[1, 1] / (3 - trR)))
    n2 = np.sqrt(abs(S[2, 2] / (3 - trR)))
    
    if n0 > n1 and n0 > n2:
        s1 = np.sign(S[0, 1] / (3 - trR))
        s2 = np.sign(S[0, 2] / (3 - trR))
        return np.pi * np.array([n0, s1 * n1, s2 * n2])
    elif n1 > n0 and n1 > n2:
        s0 = np.sign(S[1, 0] / (3 - trR))
        s2 = np.sign(S[1, 2] / (3 - trR))
        return np.pi * np.array([s0 * n0, n1, s2 * n2])
    else:
        s0 = np.sign(S[2, 0] / (3 - trR))
        s1 = np.sign(S[2, 1] / (3 - trR)) 
        return np.pi * np.array([s0 * n0, s1 * n1, n2])

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
