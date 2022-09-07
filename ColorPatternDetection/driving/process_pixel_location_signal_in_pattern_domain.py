import numpy as np
import matplotlib.pyplot as plt
import cv2
from reprojection import calc_square_3d

path = '/Users/oshrihalimi/Downloads/488.npy'
save_path = '/Users/oshrihalimi/Downloads/488.png'
if __name__ == '__main__':
    board_size = [300, 900] # for small pattern

    data = np.load(path, allow_pickle=True).item()
    board_location_dict = data['board_location'] # key = keypoint ID, value = board location
    pixel_location_dict = data['image_location'] # key = keypoint ID, value = pixel location

    pts = -np.ones((board_size[0], board_size[1], 2))
    for k, v in data['board_location'].items():
        pts[int(v[0]), int(v[1]), :] = np.flip(data['image_location'][k])
    # pts holds the 2D pixel location signal in the pattern domain, and (-1, -1) if there's no detection
    # plt.imshow(pts[:, :, 1], cmap='gray')
    # plt.show()

    valid_reg_mask = 1 * np.all(pts != -1, axis=2)
    valid_square = valid_reg_mask[:-1, :-1] * valid_reg_mask[:-1, 1:] * valid_reg_mask[1:, 1:] * valid_reg_mask[1:, :-1]
    cv2.imwrite(save_path, valid_square * 255)

    square_location = np.argwhere(valid_square==1)
    square_corners = np.stack((square_location,
                               square_location + np.array([1, 0]),
                               square_location + np.array([1, 1]),
                               square_location + np.array([0, 1])), axis=1) # corners are ordered cyclically
    square_corners_pixel = pts[square_corners[:, :, 0], square_corners[:, :, 1], :]
    square_corners_3d = []
    for corners in square_corners_pixel:
        square_corners_3d.append(calc_square_3d(corners))
    print("Hi")
