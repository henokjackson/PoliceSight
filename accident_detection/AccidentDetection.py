import os
import threading
import cv2 as cv
import numpy as np
from time import time,sleep
from utils.datasets import letterbox
from detect import YOLODetect,Load_YOLO,Initialize
from DenseOpticalFlow_Video import VideoSetup,VideoFrameExtract,DenseOpticalFlow
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
weights='/home/henok/Documents/B.Tech/S7/Courses/CSD415 - Final Year Project/Code/Enhancement-of-Public-Safety-using-Computer-Vision-and-NLP/fire_detection/models/best.pt'

w1=2
w2=1
vid_length=3
detected_frames=0
frame_count_threshold=5

def Resize_Frame(frame,scale_percent):

    #Downscale The Frame
    frame=cv.resize(frame,(int(frame.shape[1]*(scale_percent/100)),int(frame.shape[0]*(scale_percent/100))),interpolation=cv.INTER_AREA)

    return frame

def ExtractFrame(vid):

    #reading the video frame by frame
    ret,frame=vid.read()

    return frame

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

def Region_Draw(mask,frame):
    
    #Detect Contours
    contours,_=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame,contours,-1,(255,0,0),2,cv.LINE_4)

    #Bounding Box List
    boxes=[]

    #Estimate Detected Contours To Rectangles
    for contour in contours:
        (x,y,w,h)=cv.boundingRect(contour)
        boxes.append((x,y,w,h))
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    return frame,boxes

def Output(in_img,fg_mask,color_mask,mask,out_img,msg1,msg2,msg3,msg4,msg5):

    #Display Input
    cv.imshow(msg1,in_img)

    #Display Foreground Mask
    cv.imshow(msg2,fg_mask)

    #Display Color Mask
    cv.imshow(msg3,color_mask)

    #Display Fused Mask
    cv.imshow(msg4,mask)

    #Display Output
    cv.imshow(msg5,out_img)

if __name__=='__main__':

    #Initialize Frame Counter
    count=0

    #Initialize Save Video Buffer
    video_buffer=[]

    #Setting Video Save Path
    sav_path=0
    

    #Setting Up Video Stream Parameters
    vidpath=input("Video File Path : ")
    vid,status,fps=VideoSetup(vidpath)
    size=(int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)))

    #Calculate Video Buffer Size
    video_buffer_max_size=fps*vid_length

    #Setting Downscale Factor(%)
    scale_percent=100-int(input("Enter Downscale Factor (in %) : "))


    #Checking If Stream Is Open
    if(status==False):
        print("Error, Failed to Open Video Stream !")
        exit(0)

    else:
        #Iterate Through Each Frame
        ret,first_frame_grey,first_frame_rgb=VideoFrameExtract(vid)
        mask=np.zeros_like(first_frame_rgb)
        mask[..., 1]=255
        prev_frame=first_frame_grey
        while(True):

            #Extract a Frame
            frame=ExtractFrame(vid)
            
            #Checking If Extracted Frame Is Valid
            if str(type(frame))!="<class 'numpy.ndarray'>":
                break

            #Keep Original Image Copy
            frame_org=frame.copy()

            #Downscaling Frame Size
            #frame=Resize_Frame(frame,scale_percent)

            _,next_frame_grey,next_frame_rgb=VideoFrameExtract(vid)
            prev_frame=first_frame_grey


            output,mag,ang=DenseOpticalFlow(mask,prev_frame,next_frame_grey)

            
            ####################################
            # Extracting Fire-Coloured Regions #
            ####################################

            #Keep Resized Frame Copy
            in_img=frame.copy()
            out_img=frame.copy()

            #Start Timer For Calculating Real-Time FPS
            start=time()


            #Stop Timer
            stop=time()


            #Counting No. Of Frames Processed
            count+=1
            
            #Printing Debug Info
            #os.system('clear')
            print("Frames Processed : "+str(count))
            print("FPS : "+str(round(1/(stop-start))))
            
            #Skip For First Frame
            if count!=1:
                #Check Type Of Randomness
                if len(frame_queue[0][2])==len(frame_queue[1][2])==1:
                    #Checking Of Current Frame Pointer Is Invalid
                    #Checking Area Randomness In Consecutive Frames
                    Area_Randomness_Check(frame_queue[0][2],frame_queue[1][2])

                elif len(frame_queue[0][2])!=0 and len(frame_queue[1][2])!=0:
                    #Checking Box Count Randomness In Consecutive Frames
                    Box_Randomness_Check(len(frame_queue[0][2]),len(frame_queue[1][2]))

                frame_queue[0]=frame_queue[1]
                frame_queue[1]=(mask,pixel,boxes)

            detected_frames=round(detected_frames)

            #Queueing Frames To Video Buffer
            if detected_frames>0:
                #Append The Frame
                video_buffer.append(frame_org)
                detected_frames=detected_frames-1
                if detected_frames==0 and len(video_buffer)>=video_buffer_max_size:
                    print("Queue Ready !")
                    #Setting Up Multithreading For VideoWriter + YOLODetect()
                    yolo_detect_thread=threading.Thread(target=YOLO,name="YOLO Thread",args=(video_buffer.copy(),fps,size,sav_path,))

                    #Starting All Threads
                    yolo_detect_thread.start()

                    #Resetting Variables
                    detected_frames=0
                    video_buffer.clear()
                    sav_path=sav_path+1

            #Displaying Input, Mask and Final Output
            Output(in_img,out_img,"CCTV Input","Foreground Mask","Color Mask","Fused Mask","Output")

            #Waiting For Keypress -> Quit OpenCV 'imshow()' Window
            if cv.waitKey(1) & 0XFF == ord('q'):
                break

        #Close Video Stream and Cleanup
        cv.destroyAllWindows()
        vid.release()

        #Print Success Message
        print("Done Parsing Frames.")