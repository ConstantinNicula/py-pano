# py-pano

An fully automated image stitching pipeline written in Python, capable of handling unordered image sets. 

### Implementation overview
The stitching pipeline's implementation draws heavily from the work of G. Lowe and Mathew Brown, as detailed in their article ['Automatic Panoramic Image Stitching using Invariant Features'](http://matthewalunbrown.com/papers/ijcv2007.pdf): 

- Feature extraction and matching: this is the first stage in the pipeline and is responsible for identifying all pairs of overlapping images. A simple O(N^2) algorithm is employed to identify these image pairs. SIFT features and descriptors are precomputed for all input images. For each unique pair of images, the extracted features are matched and a Homography which best explains the relative motion is estimated. A geometric validation heuristic based on the number of inlier and outlier features is used to decide if an image pair contains a valid overlap or not.

- Bundle Adjustment: determines a globally consistent set of camera poses (rotations) and focal lengths. Certain features are observed in multiple reference images, thus, it is necessary to find a set of camera parameters which accounts for this. This essentially boils down to a nonlinear optimization problem in which the parameters are the camera poses and the error metric is the sum of reprojection errors. The implementation leverages LM optimizer available in SciPy and a few tricks to speed up convergence (initialization extrapolated from Homographies and analytically computed Jacobians).

- Compositing: once the camera poses are available, the input images can be transformed into a common reference frame and stitched together. Multi-band blending is used in order to avoid jarring transitions caused by misalignments between overlapping images.   

## Setup and running
The code in this repo requires Python >=3.9 and relies on a series of standard packages. To simplify installation a `requirements.txt` is provided in the root directory of the repo.  

The following steps are required to setup and run the demo: 
- setup conda / virtualenv with with Python >=3.9 
- `pip install -r requirements.txt` - install python dependencies 
- `python main.py <path_to_input>` - create panorama for images in provided folder, output is stored in <path_to_input>/out/out.png

## Examples
![](https://github.com/ConstantinNicula/py-pano/blob/main/input_imgs/ntulib/out/out.png)
![](https://github.com/ConstantinNicula/py-pano/blob/main/input_imgs/hanafish/out/out.png)
![](https://github.com/ConstantinNicula/py-pano/blob/main/input_imgs/apron/out/out.png)
Test images taken from this website: http://www.photofit4panorama.com/gallery.html


