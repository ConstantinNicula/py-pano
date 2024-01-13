import numpy as np
import cv2 


def scale_image(img: np.ndarray, max_target_dim: int) -> tuple[np.ndarray, float]:
    # Compute target scale
    h, w, _ = img.shape
    
    # Compute new w, h
    s = (max_target_dim) / max(h, w)

    # Early exit if scale is below 
    if s > 1.0: return (img, 1)
    new_h = int(round(s * h))    
    new_w = int(round(s * w))    

    return (cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC), s) 

def image_center(img: np.ndarray, scale: float = 1) -> np.ndarray:
    h, w, _ = img.shape
    return scale * np.array([(w-1)/2, (h-1)/2])

def get_width_height(img: np.ndarray) -> tuple[int, int]:
    return img.shape[1], img.shape[0]