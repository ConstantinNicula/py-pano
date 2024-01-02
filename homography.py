import numpy as np

def estimate_homography_ransac(kpts1: np.ndarray, kpts2: np.ndarray, reproj_err=5, max_iters=500) -> tuple[np.ndarray, np.ndarray]:
    # Implementation based on: 
    # https://engineering.purdue.edu/kak/courses-i-teach/ECE661.08/solution/hw4_s1.pdf
    H, inlier_mask = None, None
    
    num_keypoints = len(kpts1)
    num_inliers = 0
    err_std = np.Inf
    err_mean = np.Inf

    i, N = 0, max_iters 
    while i < N:
        # extract 4 potential inliers at random
        sel_pairs = np.random.choice(num_keypoints, 4, replace=False)
        sel_kpts1 = kpts1[sel_pairs]
        sel_kpts2 = kpts2[sel_pairs]

        # estimate homography from 4 pairs 
        Hi = estimate_homography(sel_kpts1, sel_kpts2) 

        # calculate the number of inliers
        cur_num_inliers, curr_std, cur_err, curr_inlier_mask = find_inliers(Hi, kpts1, kpts2, reproj_err) 
        
        # update model if necessary
        if cur_num_inliers > max(4, num_inliers) or (cur_num_inliers == num_inliers and curr_std < err_std):
            # store best model
            num_inliers, err_std, err_mean, inlier_mask = cur_num_inliers, curr_std, cur_err, curr_inlier_mask
            H = Hi
        i += 1
    
    # improve estimate using all points
    if H is not None: 
        H = refine_estimate(kpts1[inlier_mask], kpts2[inlier_mask])
        num_inliers, err_std, err_mean, inlier_mask = find_inliers(H, kpts1, kpts2, reproj_err)
    return H, inlier_mask

def refine_estimate(kpts1: np.ndarray, kpts2: np.ndarray) -> np.ndarray:
    cond_kpts1, T1 = normalize_input(kpts1)
    cond_kpts2, T2 = normalize_input(kpts2)

    H = estimate_homography(cond_kpts1, cond_kpts2)
    H = np.linalg.inv(T2) @ H @ T1
    H /= H[-1, -1]
    return H

def normalize_input(kpts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # convert to x,y from homogenous
    xy = kpts[:, 0:2]

    # compute center
    t = np.mean(xy, axis=0)

    # compute distances & scaling factor
    dist = np.linalg.norm(xy - t, axis=1)
    s = np.sqrt(2) / np.mean(dist, axis=0)
    cond_mat = np.array([
        [s, 0, -s*t[0]],
        [0, s, -s*t[1]],
        [0, 0,       1]
    ])

    cond_pts = transform_points(cond_mat, kpts)
    return cond_pts, cond_mat

def estimate_homography(kpts1: np.ndarray, kpts2: np.ndarray) -> np.ndarray:
    N = len(kpts1)
    assert len(kpts1) == len(kpts2) and N >= 4

    A = np.zeros((2 * N, 9))
    for i in range(N):
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

def find_inliers(H: np.ndarray, kpts1: np.ndarray, kpts2: np.ndarray, reproj_err: float) -> tuple[int, float, float, np.ndarray]: 
    # compute transformed points
    pt_est = (H @ kpts1.T).T
    pt_est = (pt_est.T / pt_est[:, 2]).T

    # compute estimation error
    err = kpts2 - pt_est
    norm_err = np.linalg.norm(err, axis=1)

    # extract inliers and the corresponding errors
    inlier_mask = norm_err < reproj_err
    errors = norm_err[inlier_mask]
    return np.count_nonzero(inlier_mask), np.std(errors), np.mean(errors), inlier_mask

def convert_to_homogenous(kpts: np.ndarray): 
    ones_col = np.ones((len(kpts), 1))
    return np.hstack((kpts, ones_col))

def transform_points(H: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    kpts_t = (H @ kpts.T).T
    return (kpts_t.T / kpts_t[:, 2]).T    
    