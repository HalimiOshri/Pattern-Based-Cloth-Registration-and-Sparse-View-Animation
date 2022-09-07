import os
import cv2
import numpy as np

camera = 401536
frame = 6210
example_image_path = f'/mnt/home/oshrihalimi/Data/ColorPattern/cam{camera}/image{frame}.png'
save_path = '/mnt/home/oshrihalimi/color_pattern_detection/'
save_dir = 'initial_tests'

if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, save_dir), exist_ok=True)

    image = cv2.imread(example_image_path)
    cv2.imwrite(os.path.join(save_path, save_dir, 'image.png'), image)

    im_negative = np.roll(image, shift=1, axis=2)
    cv2.imwrite(os.path.join(save_path, save_dir, 'negative_image.png'), im_negative)

    im_negative_2 = np.roll(image, shift=2, axis=2)
    cv2.imwrite(os.path.join(save_path, save_dir, 'negative_image_2.png'), im_negative_2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(save_path, save_dir, 'gray.png'), gray[..., None])

    gray_negative = cv2.cvtColor(im_negative, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(save_path, save_dir, 'gray_negative.png'), gray_negative[..., None])

    gray_negative_2 = cv2.cvtColor(im_negative_2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(save_path, save_dir, 'gray_negative_2.png'), gray_negative_2[..., None])

    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    cv2.imwrite(os.path.join(save_path, save_dir, 'blurred.png'), blurred)

    blurred_negative = cv2.GaussianBlur(gray_negative, (5, 5), 2)
    cv2.imwrite(os.path.join(save_path, save_dir, 'blurred_negative.png'), blurred_negative)

    blurred_negative_2 = cv2.GaussianBlur(gray_negative_2, (5, 5), 2)
    cv2.imwrite(os.path.join(save_path, save_dir, 'blurred_negative.png'), blurred_negative_2)

    #laplacian = np.stack([cv2.Laplacian(blurred[:, :, i], cv2.CV_64F, ksize=11) for i in range(3)], axis=2)
    #laplacian = np.linalg.norm(laplacian, axis=2, keepdims=True)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=5)
    laplacian_negative = cv2.Laplacian(blurred_negative, cv2.CV_64F, ksize=5)
    laplacian_negative_2 = cv2.Laplacian(blurred_negative_2, cv2.CV_64F, ksize=5)

    laplacian_mask = ((laplacian + laplacian_negative + laplacian_negative_2) > 50).astype(np.uint8)

    cv2.imwrite(os.path.join(save_path, save_dir, 'laplacian_and_image.png'), image * laplacian_mask[:, :, None])
    cv2.imwrite(os.path.join(save_path, save_dir, 'laplacian.png'), 255 * laplacian_mask)


    # 1
    islands = 1 - laplacian_mask
    cv2.imwrite(os.path.join(save_path, save_dir, 'islands.png'), 255 * islands)
    kernel = np.ones((3, 3), np.uint8)
    output = cv2.connectedComponentsWithStats(islands.astype(np.uint8), 4, cv2.CV_8U)
    (numLabels, labels, stats, centroids) = output
    radius = 1
    thickness = 1
    color = (0, 255, 0)
    for i in range(centroids.shape[0]):
        image = cv2.circle(image, (int(centroids[i][0]), int(centroids[i][1])), radius, color, thickness)

    # # 2
    # islands = cv2.morphologyEx(islands, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite(os.path.join(save_path, save_dir, 'islands_2.png'), 255 * islands)
    # output = cv2.connectedComponentsWithStats(islands.astype(np.uint8), 4, cv2.CV_8U)
    # (numLabels, labels, stats, centroids) = output
    #
    # color = (255, 0, 0)
    # for i in range(centroids.shape[0]):
    #     image = cv2.circle(image, (int(centroids[i][0]), int(centroids[i][1])), radius, color, thickness)
    #
    # # 3
    # islands = cv2.morphologyEx(islands, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite(os.path.join(save_path, save_dir, 'islands_3.png'), 255 * islands)
    # output = cv2.connectedComponentsWithStats(islands.astype(np.uint8), 4, cv2.CV_8U)
    # (numLabels, labels, stats, centroids) = output
    #
    # color = (0, 0, 255)
    # for i in range(centroids.shape[0]):
    #     image = cv2.circle(image, (int(centroids[i][0]), int(centroids[i][1])), radius, color, thickness)
    #
    # # 4
    # islands = cv2.morphologyEx(islands, cv2.MORPH_OPEN, kernel, iterations=3)
    # cv2.imwrite(os.path.join(save_path, save_dir, 'islands_4.png'), 255 * islands)
    # output = cv2.connectedComponentsWithStats(islands.astype(np.uint8), 4, cv2.CV_8U)
    # (numLabels, labels, stats, centroids) = output
    #
    # color = (0, 255, 255)
    # for i in range(centroids.shape[0]):
    #     image = cv2.circle(image, (int(centroids[i][0]), int(centroids[i][1])), radius, color, thickness)
    # cv2.imwrite(os.path.join(save_path, save_dir, f'detected_sqaures_camera_{camera}_frame_{frame}.png'), image)
    #
    # # 5
    # islands = cv2.morphologyEx(islands, cv2.MORPH_OPEN, kernel, iterations=3)
    # cv2.imwrite(os.path.join(save_path, save_dir, 'islands_5.png'), 255 * islands)
    # output = cv2.connectedComponentsWithStats(islands.astype(np.uint8), 4, cv2.CV_8U)
    # (numLabels, labels, stats, centroids) = output
    #
    # color = (255, 255, 255)
    # for i in range(centroids.shape[0]):
    #     image = cv2.circle(image, (int(centroids[i][0]), int(centroids[i][1])), radius, color, thickness)

    cv2.imwrite(os.path.join(save_path, save_dir, f'detected_sqaures_camera_{camera}_frame_{frame}.png'), image)
    print("Hi")
