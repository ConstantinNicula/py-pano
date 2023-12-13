import cv2 
import time
import numpy as np
import itertools

class Matcher: 
    def __init__(self, kpts_per_img:int = 2000): 
        # Params described https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
        self.kp_detector = cv2.ORB.create(nfeatures=kpts_per_img, scaleFactor=1.4)
        self.kp_matcher = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)

        # Contains concatenated data
        self.all_descriptors = None
        self.all_keypoints = None 
        self.all_matches = None

    """
        Find overlaps between input images and returns a set of matching points.
        Note: 
            A given ray direction can occur in multiple images, a set of matching points would than be: 
                p_i(img_n) -> (p_j(img_m), p_k(img_o)...)
            These matches have to be unique  
    """
    def match(self, imgs: list[np.ndarray], k:int = 4): 
        # preprocess all images
        self.all_descriptors = []
        self.all_keypoints = []
        for img in imgs: 
            kpts, desc = self.__get_keypoints(img)
            self.all_descriptors.append(desc)
            self.all_keypoints.append(kpts)

        # loop through images and find best matches 
        self.all_matches = []
        for i, target_desc in enumerate(self.all_descriptors): 
            ref_descriptors = self.all_descriptors[:]
            ref_descriptors.pop(i)

            self.kp_matcher.clear()
            self.kp_matcher.add(ref_descriptors)
            matches = self.kp_matcher.knnMatch(target_desc, k=4)

            # flatten all matches
            matches = tuple(itertools.chain.from_iterable(matches))
            self.all_matches.append(matches)

            # loop through all matches and retain those for m candidate images
            matched_images = [m.imgIdx for m in matches]
            matches_per_image =  {i:matched_images.count(i) for i in range(len(imgs))}
            
            # retain top m images 

            # geometric validation using RANSAC 

            print(matches_per_image) 

    def __get_keypoints(self, img: np.ndarray): 
        # convert image to gray scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # perform knn match         
        kpts, desc = self.kp_detector.detectAndCompute(img_gray, None)  
        return kpts, desc 


    def __find_kp_matches(self, img1: np.ndarray, img2: np.ndarray):
        kpts1, desc1 = self.__get_keypoints(img1)
        kpts2, desc2 = self.__get_keypoints(img2)

        self.kp_matcher = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)
        self.kp_matcher.add([desc2])
        matches = self.kp_matcher.knnMatch(desc1, k=2)
        print(">>>>", matches[0][0].imgIdx)
        good_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < 0.7 * m[1].distance:
                good_matches.append((m[0],))

        # to do some filtering
        return kpts1, kpts2, good_matches 

    def __utils_resize(self, img: np.ndarray, scale = 0.5):
        new_dims = (int(img.shape[1] * scale), int(img.shape[0] * scale)) 
        return cv2.resize(img, new_dims)

    def test_extract(self, img, scale = 0.5):
        img = self.__utils_resize(img, scale) 
        kpts, desc = self.__get_keypoints(img)
        print(f"Detected {len(kpts)} keypoints")
        print(len(desc[0]), kpts[0]) 
        return
        # show result
        img_disp = img.copy()
        cv2.drawKeypoints(img, kpts, img_disp)
        cv2.imshow("Detected keypoints", img_disp)
        cv2.waitKeyEx(0)
        cv2.destroyAllWindows()

    def test_match(self, img1, img2, scale = 0.5):
        img1 = self.__utils_resize(img1, scale)
        img2 = self.__utils_resize(img2, scale)

        kpts1, kpts2, match = self.__find_kp_matches(img1, img2)
        img_out = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2, match, None)
        cv2.imshow("matches", img_out)
        cv2.waitKey(0)


import loader
if __name__ == "__main__":
    imgs, exif = loader.load_batch("./input_imgs/apron")
    matcher = Matcher()
    t1 = time.time()
    matcher.match(imgs)
    print(f"Delta time {time.time() - t1}")
    #matcher.test_extract(imgs[0])
    #res = matcher.test_match(imgs[0], imgs[2])
