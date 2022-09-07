import cv2
import os
texture_size = 1024
path_textures = '/mnt/home/oshrihalimi/compare_methods_paper/optical_flow/donglai_without_pattern_registration/clothesUVTexture1024/'
save_path = '/mnt/home/oshrihalimi/compare_methods_paper/optical_flow/donglai_without_pattern_registration/clothesUVTexture1024/'
start_frame = 1000
end_frame = 1500

if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)
    video = cv2.VideoWriter(os.path.join(save_path, f"optical_flow_texture_frame_{start_frame}_{end_frame}.mp4"), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (texture_size, texture_size))

    for frame in range(start_frame + 1, end_frame):
        texture_filename_prev = f'{frame-1:04d}_median.png'
        texture_image_prev = cv2.imread(os.path.join(path_textures, texture_filename_prev))
        gray_prev = cv2.cvtColor(texture_image_prev, cv2.COLOR_BGR2GRAY)

        texture_filename = f'{frame:04d}_median.png'
        texture_image = cv2.imread(os.path.join(path_textures, texture_filename))
        gray = cv2.cvtColor(texture_image, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        video.write(texture_image)

    cv2.destroyAllWindows()
    video.release()