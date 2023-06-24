import cv2
import numpy as np
import os
import time
import threading
from collections import deque
from Detect import YOLODetect,Load_YOLO,Initialize
from utils.datasets import letterbox

global w1
global w2
global half
global model
global device
global loaded
global weights
global vid_length
global detected_frames
global frame_count_threshold

half=None
model=None
device=None
loaded=False
weights='/home/henok/Documents/B.Tech/S7/Courses/CSD415 - Final Year Project/Code/Enhancement-of-Public-Safety-using-Computer-Vision-and-NLP/accident_detection/models/best.pt'
w1=2
w2=1
vid_length=3
detected_frames=0
frame_count_threshold=5


def YOLO(video_buffer,fps,size,sav_path):
    global half
    global model
    global device
    global loaded
    global weights

    #Checking GPU Variables
    if loaded==False:
        device,half=Initialize()
        model=Load_YOLO(weights,device)
        loaded=True
    
    #Load YOLO Model
    dataset=[]
    if device!=None and half!=None and loaded==True:
        for frame in video_buffer:
            #YOLO Image Formatter
            img0=frame
            img=letterbox(img0,640,32)[0]
            img=img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            dataset.append((img,img0))

        #YOLO Detector
        YOLODetect(dataset,weights,model,half,device,fps,size,sav_path,video_buffer)

def process_detected_frames(frames,fps,size,sav_path):
    print("\nNumber of Frames detected:")
    print(len(frames))
    print("\nFPS:")
    print(fps)
    os.makedirs("accident_frames", exist_ok=True)
    for i, buffered_frame in enumerate(frames):
                cv2.imwrite(f"accident_frames/frame_{i}.jpg", buffered_frame)
    input("Press Enter to continue...")
    print("\nQueue Ready !")
    #Setting Up Multithreading For VideoWriter + YOLODetect()
    yolo_detect_thread=threading.Thread(target=YOLO,name="YOLO Thread",args=(frames.copy(),fps,size,sav_path,))
    #Starting All Threads
    yolo_detect_thread.start()

    


def resize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


if __name__=='__main__':
    
    frame_index = 0

    #Setting Up Video Stream Parameters
    vidpath=input("Video File Path : ")
    video = cv2.VideoCapture(vidpath)
    size=(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ret, old_frame = video.read()
    scale_percent=100-int(input("Enter Downscale Factor (in %) : "))
    if ret:
        old_frame = resize_frame(old_frame, scale_percent)
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    frame_buffer = deque(maxlen=91)

    magnitude_buffer = deque(maxlen=5)

    save_extra_frames = False
    extra_frames_counter = 0

    prev_time = time.time()
    #Setting Video Save Path
    sav_path=0

    detected_frames = []
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            frame = resize_frame(frame, scale_percent)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            os.system('cls' if os.name == 'nt' else 'clear')

            print(f"FPS: {fps}")

            original_frame = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate Dense Optical Flow
            flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Compute Magnitude and Angle of 2D Vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Normalize Magnitude from 0 to 255 and calculate its Average
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            avg_magnitude = np.mean(magnitude)

            # Calculate the difference in Average Magnitude between the Current Frame and the Previous Frame
            magnitude_diff = avg_magnitude - magnitude_buffer[-1] if magnitude_buffer else 0

            magnitude_buffer.append(avg_magnitude)

            avg_magnitude_5_frames = np.mean(magnitude_buffer)

            # Create a Binary Mask where the current Average Magnitude is significantly above or below the Average Magnitude of the last 5 Frames
            motion_mask = np.where((magnitude > avg_magnitude_5_frames * 100) | (magnitude_diff > avg_magnitude_5_frames * 1), 255, 0).astype(np.uint8)

            # Find the contours of the moving regions
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > 500 and cv2.isContourConvex(cnt) == False:
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)

            if contours and not save_extra_frames:
                detected_frames = list(frame_buffer)
                if(len(detected_frames)==91):
                    save_extra_frames = True
                    process_detected_frames(detected_frames,fps,size,sav_path)
                    sav_path=sav_path+1
                    detected_frames.clear()

            frame_buffer.append(original_frame)

            if save_extra_frames:
                extra_frames_counter += 1

                if extra_frames_counter == 91:
                    save_extra_frames = False
                    extra_frames_counter = 0

            old_gray = gray.copy()
            frame_index += 1

        else:
            break

    video.release()
    cv2.destroyAllWindows()
