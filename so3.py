import numpy as np

# Implementation references: 
# https://cvg.cit.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf

def exp(phi: np.ndarray) -> np.ndarray: 
    t = np.sqrt(np.dot(phi, phi))
    ct, st = np.cos(t), np.sin(t)
    n = phi / t if t != 0.0 else np.array([1.0, 0.0, 0.0])
    return ct * np.eye(3) + (1 - ct) * np.outer(n, n)  + st * hat(n) 

def log(R: np.ndarray) -> np.ndarray:
        cos_angle = np.clip(0.5 * np.trace(R) - 0.5, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if np.isclose(angle, 0.): 
            return vee(R - np.identity(3))
        return vee((0.5 * angle / np.sin(angle)) * (R - R.T))

def hat(x: np.ndarray) -> np.ndarray:
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])

def vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]])