import numpy as np

def estimate_homography_ransac(kpts1: np.ndarray, kpts2: np.ndarray, reproj_err=5, max_iters=500) -> tuple[np.ndarray, np.ndarray]:
    # Implementation based on: 
    # https://engineering.purdue.edu/kak/courses-i-teach/ECE661.08/solution/hw4_s1.pdf
    H, inlier_mask = None, None
    
    num_keypoints = len(kpts1)
    max_num_inliers = 0
    err_std = np.Inf

    i, N = 0, max_iters 
    while i < N:
        # extract 4 potential inliers at random
        sel_pairs = np.random.choice(num_keypoints, 4, replace=False)
        sel_kpts1 = kpts1[sel_pairs]
        sel_kpts2 = kpts2[sel_pairs]

        # estimate homography from 4 pairs 
        Hi = estimate_homography(sel_kpts1, sel_kpts2) 

        # calculate the number of inliers
        cur_num_inliers, curr_std, curr_inlier_mask = find_inliers(Hi, kpts1, kpts2, reproj_err) 
        
        # update model if necessary
        if cur_num_inliers > max_num_inliers or (cur_num_inliers == max_num_inliers and curr_std < err_std):
            # store best model
            H = Hi
            max_num_inliers, err_std, inlier_mask = cur_num_inliers, curr_std, curr_inlier_mask
        
        # update N            
        e = 1 - cur_num_inliers / num_keypoints + 1e-9
        N = min(max_iters, int(np.log(1 - 0.99)/ np.log(1 - (1 - e)**4)))

        i += 1
    return H, inlier_mask

def convert_to_homogenous(kpts: np.ndarray): 
    ones_col = np.ones((len(kpts), 1))
    return np.hstack((kpts, ones_col))

def estimate_homography(kpts1: np.ndarray, kpts2: np.ndarray) -> np.ndarray:
    assert len(kpts1) == len(kpts2) and len(kpts1) == 4

    A = np.zeros((8, 9))
    for i in range(4):
        pta, ptb = kpts1[i], kpts2[i]
        
        # xi, yi, 1, 0, 0, 0, -xi' * xi, -xi' * yi, -xi' 
        A[2*i, 0:3] = pta 
        A[2*i, 6:9] = -ptb[0] * pta 

        # 0, 0, 0, xi, yi, 1, -yi' * xi, -yi' *yi, -yi' 
        A[2*i+1, 3:6] = pta 
        A[2*i+1, 6:9] = -ptb[1] * pta 

    # Use SVD to find null space 
    AtA = A.T @ A 
    U, S, V = np.linalg.svd(AtA)

    # Extract solution from last column (normalized)
    return np.reshape(V[-1, :], (3, 3)) / V[-1,-1] 

def find_inliers(H: np.ndarray, kpts1: np.ndarray, kpts2: np.ndarray, reproj_err: float) -> tuple[int, float, np.ndarray]: 
    # compute transformed points
    pt_est = (H @ kpts1.T).T
    pt_est = (pt_est.T / pt_est[:, 2]).T

    # compute estimation error
    err = kpts2 - pt_est
    norm_err = np.linalg.norm(err, axis=1)

    # extract inliers and the corresponding errors
    inlier_mask = np.flatnonzero(norm_err < reproj_err)
    errors = norm_err[inlier_mask]

    return len(inlier_mask), np.std(errors), inlier_mask
