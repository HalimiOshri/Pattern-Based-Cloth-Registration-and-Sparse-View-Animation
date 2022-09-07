import cv2
import numpy as np

cloth_mask_path = '/mnt/home/oshrihalimi/compare_methods_paper/donglai_without_pattern_registration/patterned_cloth_pose2clothes_temporalConv_untrackedBody_codec30k_icp_registration/temporal_skip_iter_160000_patterned_cloth_test_collision_resolved/clothes_mask400143/image{frame:06d}.png'
cloth_texture_path = '/mnt/home/oshrihalimi/compare_methods_paper/donglai_without_pattern_registration/patterned_cloth_pose2clothes_temporalConv_untrackedBody_codec30k_icp_registration/temporal_skip_iter_160000_patterned_cloth_test_collision_resolved/clothesDiffRender400143/image{frame:06d}.png'
body_texture_path = '/mnt/home/oshrihalimi/compare_methods_paper/donglai_without_pattern_registration/patterned_cloth_pose2clothes_temporalConv_untrackedBody_codec30k_icp_registration/temporal_skip_iter_160000_patterned_cloth_test_collision_resolved/bodyDiffRender400143/image{frame:06d}.png'
video_save_path = '/mnt/home/oshrihalimi/compare_methods_paper/donglai_without_pattern_registration/patterned_cloth_pose2clothes_temporalConv_untrackedBody_codec30k_icp_registration/temporal_skip_iter_160000_patterned_cloth_test_collision_resolved/rendering_video.mp4'
if __name__ == '__main__':
    video = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (1334, 2048))

    for i in range(0, 450):
        cloth_mask = 1 * (cv2.imread(cloth_mask_path.format(frame=i)) == 255)
        cloth = cv2.imread(cloth_texture_path.format(frame=i))
        body = cv2.imread(body_texture_path.format(frame=i))
        frame = cloth_mask * cloth + (1 - cloth_mask) * body
        video.write(frame.astype(np.uint8))

    cv2.destroyAllWindows()
    video.release()