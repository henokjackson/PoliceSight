import os
from time import time
import cv2 as cv
import numpy as np
from Color_Detection_YCbCr import Localize_Fire
def Setup_Parameters(fps):
    
    #Perform morphologcal processing
    #Setup kernel - circular kernel
    '''
    kernel=np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]],dtype=np.uint8)
    '''

    #Dilation (higher iterations - more larger blobs)
    #mask=cv.dilate(mask,kernel,iterations=2)
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))

    #Initialize the background subtractor
    bg_subtractor=cv.createBackgroundSubtractorKNN(history=2, detectShadows=False)

    return kernel,bg_subtractor

def Bg_Subtract(curr_frame,bg_subtractor):
   
    # Apply background subtraction to the frame
    fg_mask=bg_subtractor.apply(curr_frame)

    # Convert the frame to grayscale
    gray=cv.cvtColor(curr_frame,cv.COLOR_BGR2GRAY)

    return fg_mask

def VideoSetup(vidpath):

    #opening video stream
    vid=cv.VideoCapture(vidpath)

    #checking stream status
    status=vid.isOpened()

    fps=vid.get(cv.CAP_PROP_FPS)

    return vid,status,abs(fps)

def ExtractFrame(vid):

    #reading the video frame by frame
    ret,frame=vid.read()

    return frame

def Resize_Frame(frame,scale_percent):
    frame=cv.resize(frame,(int(frame.shape[1]*(scale_percent/100)),int(frame.shape[0]*(scale_percent/100))),interpolation=cv.INTER_AREA)
    return frame

def Output(in_img,out_img):

    #Display input
    cv.imshow('CCTV 1 Stream Input',in_img)

    #Display input
    cv.imshow('CCTV 1 Stream Ouput',out_img)

if __name__=="__main__":

    #frame counter
    count=0
    
    #setting up video stream parameters
    vidpath=input("Video File Path : ")
    vid,status,fps=VideoSetup(vidpath)

    #setup scale factor
    scale_percent=100-int(input("Enter Downscale Factor (in %) : "))

    #checking if video stream is open
    if(status==False):
        print("Error, Failed to Open Video Stream !")
        exit(0)
    else:

        #setting up
        kernel,bg_subtractor=Setup_Parameters(fps)

        #iterating through each frame of the stream
        while(True):

            #extract frame
            frame=ExtractFrame(vid)
            
            #checking if the extracted frame is valid
            if str(type(frame))!="<class 'numpy.ndarray'>":
                break

            frame=Resize_Frame(frame,scale_percent)

            #localize fire
            in_img=frame.copy()
            start=time()
            
            #Remove Noise
            frame=cv.GaussianBlur(frame,(5,5),1)
            
            #Generate Foreground Mask
            mask1=Bg_Subtract(frame,bg_subtractor)

            #Generate Color Mask
            mask2,_,_,_=Localize_Fire(frame)

            #Output(mask1,mask2)

            mask=np.zeros_like(mask1)

            #Calculate Intersection of Two Masks
            for x in range(mask1.shape[0]):
                for y in range(mask2.shape[1]):
                    if mask1[x][y]==mask2[x][y]:
                        mask[x][y]=mask1[x][y]
                    else:
                        mask[x][y]=0

            #mask=np.intersect1d(mask1,mask2)

           
            if mask.size!=0:
                #Perform Morphological Processing
                mask=cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)
                #Detect contours
                contours,_=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
                #cv.drawContours(frame,contours,-1,(255,0,0),2,cv.LINE_4)
                
                #Estimate contours to rectangles
                for contour in contours:
                    (x,y,w,h)=cv.boundingRect(contour)
                    cv.rectangle(in_img,(x,y),(x+w,y+h),(0,0,255),2)
            else:
                mask=np.zeros_like(mask1)
            stop=time()

            #displaying output
            Output(in_img,mask)

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