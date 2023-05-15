import cv2
import numpy as np
import os
import time
from collections import deque

def resize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

os.makedirs("accident_frames2", exist_ok=True)
frame_index = 0

video = cv2.VideoCapture('test/2.mp4')
ret, old_frame = video.read()

if ret:

    old_frame = resize_frame(old_frame, 100) 
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

frame_buffer = deque(maxlen=5)

magnitude_buffer = deque(maxlen=5)

save_extra_frames = False
extra_frames_counter = 0

prev_time = time.time()

while(video.isOpened()):
    ret, frame = video.read()

    if ret:

        frame = resize_frame(frame, 100)  

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        os.system('cls' if os.name == 'nt' else 'clear')

        print(f"FPS: {fps}")

        original_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Calculate Dense Optical Flow
        flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        #Compute Magnitude and Angle of 2D Vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        #Normalize Magnitude from 0 to 255 and calculate its Average
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        avg_magnitude = np.mean(magnitude)
        
        #Calculate the difference in Average Magnitude between the Current Frame and the Previous Frame
        magnitude_diff = avg_magnitude - magnitude_buffer[-1] if magnitude_buffer else 0
        
        magnitude_buffer.append(avg_magnitude)
        
        avg_magnitude_5_frames = np.mean(magnitude_buffer)

        #Create a Binary Mask where the current Average Magnitude is significantly above or below the Average Magnitude of the last 5 Frames
        motion_mask = np.where((magnitude > avg_magnitude_5_frames * 100) | (magnitude_diff > avg_magnitude_5_frames * 1), 255, 0).astype(np.uint8)
        
        #Find the contours of the moving regions
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 500 and cv2.isContourConvex(cnt) == False: 
                cv2.drawContours(frame, [cnt], -1, (0,255,0), 3)

        # if any significant motion is detected
        if contours and not save_extra_frames:

            for i, buffered_frame in enumerate(frame_buffer):
                cv2.imwrite(f"accident_frames2/frame_{frame_index - 5 + i}.jpg", buffered_frame)
            
            save_extra_frames = True

        frame_buffer.append(original_frame)
        
        if save_extra_frames:
            
            cv2.imwrite(f"accident_frames2/frame_{frame_index}.jpg", original_frame)
            extra_frames_counter += 1
            
            if extra_frames_counter == 5:
                save_extra_frames = False
                extra_frames_counter = 0

        old_gray = gray.copy()
        frame_index += 1
        
    else:
        break

video.release()
cv2.destroyAllWindows()
