import numpy as np
import time

def estimate_homography_ransac(kpts1: np.ndarray, kpts2: np.ndarray, reproj_err=5, max_iters=500, batch_size=50) -> tuple[np.ndarray, np.ndarray]:
    # Implementation based on: 
    # https://engineering.purdue.edu/kak/courses-i-teach/ECE661.08/solution/hw4_s1.pdf
    num_keypoints = len(kpts1)

    # Solution parameters
    H = None
    inlier_mask = None
    num_inliers, err_std, err_mean = -1, np.Inf, np.Inf

    i, N = 0, max_iters
    while i < N:
        # 0) Select N batches of 4 points at random
        sel_pairs = np.array([np.random.choice(num_keypoints, 4, replace=False) for _ in range(batch_size)])

        # 1) Extract corresponding points
        sel_kpts1 = kpts1[sel_pairs]
        sel_kpts2 = kpts2[sel_pairs]

        # 2) Estimate all homographies at once
        H_batch = estimate_homography(sel_kpts1, sel_kpts2)

        # 3) Find inlier and stats for all matrices 
        batch_num_inliers, batch_std, batch_err, batch_inlier_mask = find_inliers(H_batch, kpts1, kpts2, reproj_err) 

        # 4) Extract best match
        best_j = -1 
        for j in range(batch_size):
            if batch_num_inliers[j] > max(4, num_inliers) or \
            (batch_num_inliers[j] == num_inliers and batch_std[j] < err_std):
                # update estimate
                best_j = j
        
        # 5) Store best estimate
        if best_j != -1:
            num_inliers = batch_num_inliers[best_j]
            err_mean = batch_err[best_j]
            err_std = batch_std[best_j]
            inlier_mask = batch_inlier_mask[best_j, :]            
            H = H_batch[best_j, :, :] 

            # 6) Update number of steps based on current estimates
            e = 1 - num_inliers / num_keypoints + 1e-9
            N = min(max_iters, int(np.log(1 - 0.99)/ np.log(1 - (1 - e)**4)))
            
        # 7) Update number of steps taken
        i += batch_size

    # improve estimate using all points
    if H is not None: 
        H = refine_estimate(kpts1[inlier_mask], kpts2[inlier_mask])
        num_inliers, err_std, err_mean, inlier_mask = find_inliers(H, kpts1, kpts2, reproj_err)
        inlier_mask = inlier_mask[0] # cast to proper size

    return H, inlier_mask


def refine_estimate(kpts1: np.ndarray, kpts2: np.ndarray) -> np.ndarray:
    cond_kpts1, T1 = normalize_input(kpts1)
    cond_kpts2, T2 = normalize_input(kpts2)

    H = estimate_homography(cond_kpts1, cond_kpts2)[0]
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
    """
        Estimate homography from point pairs. A minimum of 4 point pairs must be provided. 
        Multiple homographies can be estimated at once if multiple stacked point pairs are provided:
        Input: 
            kpts1 - (M, N, 3) 
            kpts2 - (M, N, 3)
        Output: 
            H - (M, 3, 3)
    """
    assert kpts1.shape == kpts2.shape and kpts1.ndim >=2
    if kpts1.ndim == 2: 
        kpts1 = np.reshape(kpts1, (1, -1, 3))
        kpts2 = np.reshape(kpts2, (1, -1, 3)) 

    # 0) make sure solution even exists
    M, N, K = kpts1.shape
    assert K == 3 and N >=4 

    # 1) Create A matrix (M, 2N, 9)
    A = np.zeros((M, 2 * N, 9))
    for i in range(N):
        # extract relevant pts (M, 3)
        pta = kpts1[:, i]
        ptb = kpts2[:, i]

        # xi, yi, 1, 0, 0, 0, -xi' * xi, -xi' * yi, -xi' 
        A[:, 2*i, 0:3] = pta 
        A[:, 2*i, 6:9] = -np.reshape(ptb[:, 0], (-1, 1)) * pta 

        # 0, 0, 0, xi, yi, 1, -yi' * xi, -yi' *yi, -yi' 
        A[:, 2*i+1, 3:6] = pta 
        A[:, 2*i+1, 6:9] = -np.reshape(ptb[:, 1], (-1, 1)) * pta 

    # 2) Compute A.T * A - (N, 9, 9) 
    # AtA = A.T @ A 
    AtA = np.transpose(A, (0, 2, 1)) @ A 

    # 3) Use SVD to find null space 
    U, S, V = np.linalg.svd(AtA)

    # 4) Extract solution from last column and normalize
    H_stack = np.reshape(V[:, -1, :], (-1, 3, 3))
    H_stack = H_stack / np.reshape(H_stack[:, -1, -1], (-1, 1, 1))
    return H_stack 


def find_inliers(H: np.ndarray, kpts1: np.ndarray, kpts2: np.ndarray, 
                 reproj_err: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """
        Compute inliers for multiple homography matrices.
        Input restrictions: 
            - H must have dimensions (M, 3, 3) thus each H[i] represents a 3x3 homography matrix     
            - kpts1 must have dimension (N, 3)
            - kpts2 must have dimension (N, 3)
        Output: 
            - inlier_count - (M, 1)
            - reprojection std - (M, 1)
            - reprojection mean - (M, 1) 
            - inlier_masks - (M, N)
    """
    # 0) broadcast keypoints to (1, N, 3)
    kpts1 = np.reshape(kpts1, (1, -1, 3))
    H = np.reshape(H, (-1, 3, 3))
    
    # 1) compute transformed points 
    # p_est - M, N, 3 
    pt_est = kpts1 @ np.transpose(H, (0, 2, 1))
    
    # Replaced pt_est = pt_est / pt_est[:, :, 2, None] to avoid NaNs
    pt_est_z = pt_est[:, :, 2, None]
    pt_est = np.divide(pt_est, pt_est_z, out=np.full(pt_est.shape, np.inf), where=pt_est_z!=0 )
    # 2) compute estimation error
    # err - M, N, 3  and norm_err = M, N, 1 
    err = kpts2 - pt_est
    norm_err = np.linalg.norm(err, axis=2)

    # 3) extract inliers and the corresponding errors
    # inlier_mask - M, N, 1, i
    inlier_mask = norm_err < reproj_err

    # 4) compute stats
    inlier_cnt = np.count_nonzero(inlier_mask, axis=1)
    inlier_std = np.std(norm_err, where=inlier_mask, axis=1)
    inlier_mean = np.mean(norm_err, where=inlier_mask, axis=1)
    return inlier_cnt, inlier_std, inlier_mean, inlier_mask


def convert_to_homogenous(kpts: np.ndarray): 
    ones_col = np.ones((len(kpts), 1))
    return np.hstack((kpts, ones_col))

def transform_points(H: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    kpts_t = kpts @ H.T
    return kpts_t / kpts_t[:, 2, None]
