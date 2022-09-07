import os

capture_path = '/mnt/captures/studies/pilots/sociopticon/Aug18/s--20210823--1323--0000000--pilot--patternCloth/undistorted/'
save_dir = '/mnt/home/oshrihalimi/capture'
if __name__ == '__main__':
    camera_folders = os.listdir(capture_path)
    camera_list_path = os.path.join(save_dir, 'camera_list.txt')
    with open(camera_list_path, 'a+') as f:
        f.truncate(0)
        for cam in camera_folders:
            if len(os.listdir(os.path.join(capture_path, cam))) == 0:
                continue
            line = f'{cam}\n'
            f.write(line)
    print("Hi")

