import numpy as np
import cv2
import os

os.makedirs("images/output", exist_ok=True)

def generate_image(width, height):
    image = np.ones((height, width), dtype=np.uint8) * 255

    # Rectangle (gray = 128)
    rect_x, rect_y = 20, 50
    rect_w, rect_h = 100, 150
    image[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = 128

    # Triangle (black = 0)
    pts = np.array([[200, 50], [250, 200], [150, 200]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], 0)

    return image

def add_gaussian_noise(image, mean=0, stddev=50):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# Generate and show original
generated = generate_image(300, 300)
cv2.imshow("Generated Image", generated)
cv2.imwrite("images/output/q1_generated_image.png", generated)

# Add noise and show
noisy = add_gaussian_noise(generated)
cv2.imshow("Noisy Image", noisy)
cv2.imwrite("images/output/q1_noisy_image.png", noisy)

# Apply Otsu threshold
_, otsu = cv2.threshold(noisy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("Otsu Threshold", otsu)
cv2.imwrite("images/output/q1_otsu_result.png", otsu)

cv2.waitKey(0)
cv2.destroyAllWindows()
