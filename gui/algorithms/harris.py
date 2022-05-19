import numpy as np
import cv2


def harris_response(image, k=0.04):
    src = np.copy(image)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    Ix = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=5)
    Ixx = cv2.GaussianBlur(src=Ix**2, ksize=(5, 5), sigmaX=0)
    Ixy = cv2.GaussianBlur(src=Iy * Ix, ksize=(5, 5), sigmaX=0)
    Iyy = cv2.GaussianBlur(src=Iy**2, ksize=(5, 5), sigmaX=0)

    det = Ixx * Iyy - (Ixy**2)
    trace = Ixx + Iyy
    harris_response_matrix = det - k * (trace**2)

    return harris_response_matrix


def corners(image, threshold=0.05):
    corners_image_matrix = np.copy(image)
    harris_response_matrix = harris_response(image)
    harris_matrix = cv2.dilate(harris_response_matrix, None)
    harris_matrix_maximum = harris_matrix.max()
    corner_indices = np.array(
        harris_matrix > (harris_matrix_maximum * threshold), dtype="int8"
    )
    corners_image_matrix[corner_indices == 1] = [0, 255, 0]
    return corners_image_matrix
