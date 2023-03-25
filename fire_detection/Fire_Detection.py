import os
import cv2 as cv
from time import time
import numpy as np
from yolov7.detect import YOLODetect
from flicker_detection.Flicker_Detection import Setup_Parameter,Flicker_Detect
from color_detection.Color_Detection_YCbCr import Localize_Fire

def VideoSetup(vidpath):

    #opening video stream
    vid=cv.VideoCapture(vidpath)

    #checking stream status
    status=vid.isOpened()

    return vid,status

def ExtractFrame(vid):

    #reading the video frame by frame
    ret,frame=vid.read()

    return frame


if __name__=="__main__":

    #frame counter
    count=0
    
    #setting up video stream parameters
    vidpath=input("Video File Path : ")
    vid,status=VideoSetup(vidpath)

    #checking if video stream is open
    if(status==False):
        print("Error, Failed to Open Video Stream !")
        exit(0)
    else:

        #Flicker Detection Intial Setup
        prev_frame,bg_subtractor,matrix_size,threshold=Setup_Parameter()

        while(True):

            #extract frame
            frame=ExtractFrame(vid)
            
            #checking if the extracted frame is valid
            if str(type(frame))!="<class 'numpy.ndarray'>":
                break

            #Log start time
            start=time()
            
            #keep a copy of every frame
            in_img=frame.copy()


            # out_img_1     ->      flicker detection output
            # out_img_2     ->      color detection output    


            #flicker detection
            prev_frame,out_img_1,fire=Flicker_Detect(frame,prev_frame,bg_subtractor,matrix_size,threshold)

            #checking if flicker was detected

            #if fire was detected
            if fire==True:
                
                #color model verification
                out_img_2,_=Localize_Fire(out_img_1)

            
            #log end time
            stop=time()
            #counting number of frames iterated
            count+=1

            #printing frame info
            os.system('clear')
            print("Frames Processed : "+str(count))
            print("FPS : "+str(round(1/(stop-start))))
 
            #waiting for quit keypress
            if cv.waitKey(1) & 0XFF == ord('q'):
                break

        #cleanup stream
        cv.destroyAllWindows()
        vid.release()

        #success
        print("Done Parsing Frames.")