import cv2 as cv
import numpy as np

# The video feed is read in as a VideoCapture object
texture_size = 1024
cap = cv.VideoCapture("/Volumes/ElementsB/Paper/videos/unwrapped_texture/texture_frame_1000_1500_pattern_registration.mp4")
save_path = "/Volumes/ElementsB/Paper/videos/unwrapped_texture/optical_flow_8000_8500_big_pattern.mp4"
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_frame = first_frame
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
prev_mask_region = prev_gray != 0
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

if __name__ == '__main__':
    sum_flow = 0
    count = 0
    # video = cv.VideoWriter(save_path, cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 1, (texture_size, texture_size))
    while(cap.isOpened()):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        # Opens a new window and displays the input frame
        try:
            cv.imshow("input", frame)
        except:
            break
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask_region = gray != 0
        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, 15, 3, 5, 1.2, 0)
        kernel = np.ones((31, 31), np.uint8) # we consider the optical flow where the OF window lies completely within a visible region - unless it produces large noise
        flow_mask = cv.erode((1 * (mask_region * prev_mask_region)).astype(np.uint8), kernel, iterations=1)
        flow = flow_mask[:, :, None] * flow

        flow_avg_pixel = np.sum(np.sqrt(np.sum(flow**2, axis=-1))) / np.sum(flow_mask)
        sum_flow = sum_flow + flow_avg_pixel
        count = count + 1
        # For visualization
        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        image = prev_frame
        for i in range(0, 1024, 10):
            for j in range(0, 1024, 10):
                image = cv.arrowedLine(image, (i, j), (np.round(i + flow[j, i][0]).astype(int), np.round(j + flow[j, i][1]).astype(int)),
                                        (0, 255, 0), 1)
        cv.imwrite(f'/Users/oshrihalimi/Downloads/{count}.png', image)
        # Opens a new window and displays the output frame
        # video.write(image)

        # Updates previous frame
        prev_gray = gray
        prev_mask_region = mask_region
        prev_frame = frame
        # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()

    avg_flow = sum_flow / count
    print(avg_flow)
