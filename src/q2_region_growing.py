# Import required libraries
import cv2
import numpy as np
import os

def show_segmentation(mask):
    cv2.imshow('Segmentation Process', mask)
    cv2.waitKey(1)

def region_growing(image, seed_points, threshold_range):
    # Create a mask to store the segmented region
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Create a queue to store the seed points
    queue = []
    for seed_point in seed_points:
        queue.append(seed_point)
    
    iteration = 0
    # Perform the region growing
    while queue:
        iteration += 1
        current_point = queue.pop(0)
        current_value = image[current_point[1], current_point[0]]
        mask[current_point[1], current_point[0]] = 255

        # Display the current state every 10 iterations
        if iteration % 10 == 0:
            show_segmentation(mask)
        
        # Explore 8-connected neighborhood
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                x = current_point[0] + i
                y = current_point[1] + j
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    neighbor_value = image[y, x]
                    if abs(int(neighbor_value) - int(current_value)) <= threshold_range:
                        if mask[y, x] == 0:
                            queue.append((x, y))
                            mask[y, x] = 255
    return mask

# Read color image
color_image = cv2.imread('images/input_image.jpg')
if color_image is None:
    print("Image not found. Make sure 'images/input_image.jpg' exists.")
    exit()

# Resize to make processing manageable
scaled_image = cv2.resize(color_image, (512, 512))

# Convert to grayscale
gray_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)

# Define seed points inside the leaf (X, Y) — manually chosen
seed_points = [(250, 300), (240, 400), (270, 200)]

# Draw seed points on the scaled color image
color_with_seeds = scaled_image.copy()
for point in seed_points:
    cv2.circle(color_with_seeds, point, radius=5, color=(0, 0, 255), thickness=-1)

# Define pixel threshold
threshold_range = 10

# Run region growing
segmented_image = region_growing(gray_image, seed_points, threshold_range)

# Save results
os.makedirs('images/output', exist_ok=True)
cv2.imwrite('images/output/q2_grayscale_leaf.jpg', gray_image)
cv2.imwrite('images/output/q2_segmented_leaf.jpg', segmented_image)
cv2.imwrite('images/output/q2_seedpoints_on_color.jpg', color_with_seeds)

# Display final segmented result
cv2.destroyAllWindows()
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
