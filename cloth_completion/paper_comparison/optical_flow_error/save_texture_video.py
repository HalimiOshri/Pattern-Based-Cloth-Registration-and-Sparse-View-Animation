import cv2
import os


texture_size = 1024
path_textures = '/mnt/home/oshrihalimi/compare_methods_paper/optical_flow/big_pattern/clothesUVTexture1024'
save_path = '/mnt/home/oshrihalimi/compare_methods_paper/optical_flow/big_pattern/clothesUVTexture1024/'
start_frame = 8000#1000
end_frame = 8500#1500

if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)
    video = cv2.VideoWriter(os.path.join(save_path, f"texture_frame_{start_frame}_{end_frame}.mp4"), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (texture_size, texture_size))

    for frame in range(start_frame, end_frame):
        texture_filename = f'{frame:04d}_median.png'
        texture_image = cv2.imread(os.path.join(path_textures, texture_filename))
        if texture_image is None:
            print(frame)
        video.write(texture_image)

    cv2.destroyAllWindows()
    video.release()


