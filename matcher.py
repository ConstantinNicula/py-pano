import cv2 
import numpy as np

class Matcher: 
    def __init__(self, kpts_per_img:int = 100): 

        # Params described https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
        self.kp_detector = cv2.ORB.create(nfeatures=kpts_per_img, scaleFactor=1.4)

        # FLAN matcher params described in: 
        # Using values from:
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm = FLANN_INDEX_LSH, 
                            table_number = 6, # 12 
                            key_size = 12, #20 
                            multi_probe_level = 1) #2
        search_params = dict(checks=50)
        self.kp_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def __extract_keypoints(self, img: np.ndarray): 
        # convert image to gray scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # perform knn match         
        kpts, desc = self.kp_detector.detectAndCompute(img_gray, None)  
        return kpts, desc 

    def __find_kp_matches(self, img1: np.ndarray, img2: np.ndarray):
        kpts1, desc1 = self.__extract_keypoints(img1)
        kpts2, desc2 = self.__extract_keypoints(img2)

        matches = self.kp_matcher.knnMatch(desc1, desc2, k=2)
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
        kpts, desc = self.__extract_keypoints(img)
        print(f"Detected {len(kpts)} keypoints")
        
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
        print(match)


import loader
if __name__ == "__main__":
    imgs, exif = loader.load_batch("./input_imgs/apron")
    matcher = Matcher()
    # matcher.test_extract(imgs[0])
    res = matcher.test_match(imgs[0], imgs[2])
