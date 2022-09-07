import cv2
import numpy as np

def get_centers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 3)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=5)
    islands = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY_INV)[1]
    output = cv2.connectedComponentsWithStats(islands.astype(np.uint8), 4, cv2.CV_8U)
    (numLabels, labels, stats, centroids) = output
    return centroids