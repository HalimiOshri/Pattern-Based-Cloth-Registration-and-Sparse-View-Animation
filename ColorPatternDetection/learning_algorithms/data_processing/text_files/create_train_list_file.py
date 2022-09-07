import os
import re

save_path = '/mnt/home/oshrihalimi/Data/ColorPatternAnnotations/CornersAndCenters/'
path_patches = '/mnt/home/oshrihalimi/Data/ColorPatternAnnotations/CornersAndCenters/patches_short_subset/'
path_annotations = '/mnt/home/oshrihalimi/Data/ColorPatternAnnotations/CornersAndCenters/annotations_short/'
centers_format = 'centers-{frame:08d}.txt'
corners_format = 'corners-{frame:08d}.txt'

if __name__ == '__main__':
    with open(os.path.join(save_path, 'train_list_short.txt'), 'a+') as f:
        f.truncate(0)

        for patch in os.listdir(path_patches):
            frame = re.split('-|\.',patch)[1]
            frame_fp = os.path.join(path_patches, patch)

            centers = centers_format.format(frame=int(frame))
            centers_fp = os.path.join(path_annotations, centers)

            corners = corners_format.format(frame=int(frame))
            corners_fp = os.path.join(path_annotations, corners)

            line = f'{frame_fp} {centers_fp} {corners_fp}\n'
            f.write(line)