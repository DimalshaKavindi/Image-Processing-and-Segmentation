# EC7212 - Computer Vision and Image Processing - Assignment 2

## Implementation Details

### Task 1: Otsu's Thresholding
1. Generates base image with Rectangle (128), Triangle (0), and background (255)
2. Adds Gaussian noise (μ=0, σ=50)
3. Applies Otsu's thresholding
4. Outputs: Original, Noisy, and Thresholded images

### Task 2: Region Growing Segmentation
1. Reads input grayscale resized image
2. Starts from seed point and grows region based on pixel similarity
3. Uses 4-connectivity for neighborhood
4. Outputs: Grayscale and Segmented images

## How to Run
1. Install dependencies: `pip install numpy opencv-python`
2. Run Otsu's thresholding: `python src/q1_otsu_thresholding.py`
3. Run Region Growing: `python src/q2_region_growing.py`