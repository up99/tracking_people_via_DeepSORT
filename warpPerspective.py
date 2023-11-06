# The calculated homography can be used to warp
# the source image to destination. Size is the
# size (width,height) of im_dst
import cv2
import numpy as np
import skimage
# h11, h12, h13, h21, h22, h23, h31, h32, h33
#H = [0.00467919462394561,-0.00106004410217054,-1.09720060345245,0.00963165705752871,-0.0706995954499526,8.52759538597223,0.000189291756286462,-0.000856961083763378,0.0301514423255732]

homography_path = "HallWayTracking/homography/001.txt"
H = np.loadtxt(homography_path, delimiter=",")
H = np.array(H).reshape(((3, 3))).astype(np.float32)

# H = H.transpose()
print(H)


cap = cv2.VideoCapture("HallWayTracking/videos/001.avi")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(5))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if cap.isOpened():
    # Read a frame
        ret, frame = cap.read()
        # converting to gray-scale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        #warped = skimage.transform.warp(frame, H)
        new_frame = cv2.warpPerspective(frame, H, (frame_width, frame_height))

        cv2.imshow('input',frame)
        cv2.imshow('output',new_frame)
        # Press q to quit.
        cv2.waitKey(0)
        # showing the image 

