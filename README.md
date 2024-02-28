# Image Stitching with OpenCV
- A 2D panorama image fusion application from automatically and semi-automatically detected primitives
- Implementation details:
  - Detect descriptors with SIFT
  - Match descriptors with FLANN or Brute Force
  - Estimate the homography with RANSAC
  - Blend the images with alpha blending, gaussian pyramid or multi-band blending
## Installation
```sh
pip install -r requirements.txt
```
## Usage
```sh
python panorama.py <chemin_image1> <chemin_image2> <chemin_output_path>
```
