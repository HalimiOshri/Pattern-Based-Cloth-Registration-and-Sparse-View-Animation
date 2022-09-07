import cv2
import os


texture_size = 1024
path_textures = '/mnt/home/donglaix/s--20210823--1323--0000000--pilot--patternCloth/clothesUVTexture1024/'
save_path = '/mnt/home/oshrihalimi/compare_methods_paper/our_registration_texture_unwrapped'
start_frame = 0
end_frame = 4450

if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)
    video = cv2.VideoWriter(os.path.join(save_path, f"texture_frame_{start_frame}_{end_frame}.mp4"), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (texture_size, texture_size))

    for frame in range(start_frame, end_frame):
        texture_filename = f'{frame:06d}_median.png'
        texture_image = cv2.imread(os.path.join(path_textures, texture_filename))

        video.write(texture_image)

    cv2.destroyAllWindows()
    video.release()


