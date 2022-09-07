import cv2 as cv
import numpy as np

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("/Users/oshrihalimi/Downloads/optical flow/icp_texture_frame_0_4450.mp4")
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_frame = first_frame
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

if __name__ == '__main__':

    while(cap.isOpened()):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        # Opens a new window and displays the input frame
        cv.imshow("input", frame)
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 7, 3, 5, 1.2, 0)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        image = prev_frame
        for i in range(0, 1024, 10):
            for j in range(0, 1024, 10):
                image = cv.arrowedLine(image, (i, j), (np.round(i + flow[j, i][1]).astype(int), np.round(j + flow[j, i][0]).astype(int)),
                                        (0, 255, 0), 1)

        # Opens a new window and displays the output frame
        cv.imshow("dense optical flow", image)
        # Updates previous frame
        prev_gray = gray
        prev_frame = frame
        # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()
