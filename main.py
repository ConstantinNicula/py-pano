import argparse
import loader
import time 
from matcher import Matcher
from bundler import Bundler,PoseVisualizer, CameraPose
from homography import *
import so3
import cv2

def downscale_image(img: np.ndarray, target_dim: int)-> np.ndarray:
    h, w, _ = img.shape
    s = target_dim / max(h, w)
    if s > 1.0: 
        return img
    else: 
        dim = (int(w * s), int(h * s))
        return cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC) 


def pose_to_RK(w: float, h: float, pose: CameraPose): 
    f = pose.f
    R = so3.exp(pose.rot).astype(np.float32)
    K = np.array([[f, 0, w/2], 
                [0, f, h/2], 
                [0, 0, 1]]).astype(np.float32)
    return R, K

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_src_folder", help="Path to the folder containing input images")
    parser.add_argument("out_folder", help="Path where result will be stored")
    args = parser.parse_args()

    imgs, imgs_exif = loader.load_batch(args.img_src_folder)
    st = time.time()

    # no need to use full scale images for pose estimation, reduce resolution
    mid_imgs = []    
    for img in imgs: 
        mid_imgs.append(downscale_image(img, 1024))

    # compute image matches
    matcher = Matcher()
    match_data = matcher.match(mid_imgs)
    print(f"Time to match: {time.time() - st}s") 

    bundler = Bundler()
    bundler.set_images(mid_imgs)
    bundler.set_match_data(match_data)
    bundler.optimize(0)


    scale = bundler.pose_graph_nodes[0].f 
    warper = cv2.PyRotationWarper("spherical", scale)

    left, right = np.inf, -np.inf 
    top, down = np.inf, -np.inf 
    for i in range(len(mid_imgs)):    
        img = mid_imgs[i]
        h, w, _ = img.shape
        pose = bundler.pose_graph_nodes[i]
        R, K = pose_to_RK(w, h, pose)
        bounds = warper.warpRoi((w, h), K, R.T)
        print(bounds)
        left = min(bounds[0], left)
        top = min(bounds[1], top)
        
        right = max(bounds[0] + bounds[2], right)
        down = max(bounds[1] + bounds[3], down)

        # print(f"width: {bounds[2] - bounds[0] + 1}") 
        # print(f"height: {bounds[1] - bounds[3] + 1}") 
    proj_w = right - left + 1
    proj_h = down - top + 1
    
    print(top, left)
    print("dims", proj_h, proj_w)

    composite = np.zeros((proj_h, proj_w, 3), dtype=mid_imgs[0].dtype)
    print(left, top)
    for i in range(len(mid_imgs)):
        img = mid_imgs[i]
        h, w, _ = img.shape 
        pose = bundler.pose_graph_nodes[i]
        R, K = pose_to_RK(w, h, pose)

        tl, warp_img1_T = warper.warp(img, K, R.T, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 1)
        print("===", tl, left, top)
        tl = (tl[0] - left, tl[1] - top)
        print(">>>>", tl)
        
        dst = composite[tl[1]: tl[1] + warp_img1_T.shape[0], tl[0]: tl[0] + warp_img1_T.shape[1]]
        res = np.where(dst != np.array([0, 0, 0]), dst, warp_img1_T)
        composite[tl[1]: tl[1] + warp_img1_T.shape[0], tl[0]: tl[0] + warp_img1_T.shape[1]] = res
        print(dst.shape, warp_img1_T.shape)
        # cv2.imshow(f"warp{i}", warp_img1_T) 
    cv2.imshow(f"composite", composite)
    PoseVisualizer.display(bundler.pose_graph_nodes, mid_imgs, [])

if __name__ == "__main__":
    main()