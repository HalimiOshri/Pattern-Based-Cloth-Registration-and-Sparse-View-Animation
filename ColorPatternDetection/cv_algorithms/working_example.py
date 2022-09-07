
import os
import cv2
import numpy as np

example_image_path = '/mnt/home/oshrihalimi/Data/ColorPattern/cam401534/image0005.png'
save_path = '/mnt/home/oshrihalimi/color_pattern_detection/'
save_dir = 'initial_tests'

if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, save_dir), exist_ok=True)

    image = cv2.imread(example_image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(save_path, save_dir, 'gray.png'), gray[..., None])

    blurred = cv2.GaussianBlur(gray, (11, 11), 3)
    cv2.imwrite(os.path.join(save_path, save_dir, 'blurred.png'), blurred[..., None])

    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
    cv2.imwrite(os.path.join(save_path, save_dir, 'laplacian.png'), 255 - 255 * (laplacian > 50))

    islands = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY_INV)[1]
    output = cv2.connectedComponentsWithStats(islands.astype(np.uint8), 4, cv2.CV_8U)
    (numLabels, labels, stats, centroids) = output

    color_path = '/mnt/home/oshrihalimi/color_pattern_detection/color_classifier/18/example_cam401534_frame_0005.png'
    predicted_color_image = cv2.imread(color_path)
    # Radius of circle
    radius = 1
    # Blue color in BGR
    # color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 1
    for i in range(centroids.shape[0]):
        color = predicted_color_image[int(centroids[i][1]), int(centroids[i][0]), :]
        image = cv2.circle(image, (int(centroids[i][0]), int(centroids[i][1])), radius, (int(color[0]), int(color[1]), int(color[2])), thickness)
    cv2.imwrite(os.path.join(save_path, save_dir, 'detected_squares_color.png'), image)


    gray_predicted_colored = cv2.cvtColor(predicted_color_image, cv2.COLOR_BGR2GRAY)
    laplacian_predicted_colored = cv2.Laplacian(gray_predicted_colored, cv2.CV_64F, ksize=5)
    cv2.imwrite(os.path.join(save_path, save_dir, 'laplacian_predicted_colored.png'), 255 - 255 * (laplacian > 50))

    video = cv2.VideoWriter(os.path.join(save_path, save_dir, f"detected_squares_color.mp4"), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 1, (2668, 4096))
    video.write(image)
    video.write(predicted_color_image)
    cv2.destroyAllWindows()
    video.release()
    print("Hi")