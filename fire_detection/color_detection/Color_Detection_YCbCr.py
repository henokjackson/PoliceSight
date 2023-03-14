import os
from time import time
import cv2 as cv
import numpy as np

def Localize_Fire(frame):

    #pixel count
    pixel=0

    #convert image from BGR to YCbCr
    #frame_rgb=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    frame_ycbcr=cv.cvtColor(frame,cv.COLOR_BGR2YCR_CB)

    canvas=np.zeros_like(frame)

    #Rule Set - I
    Y_mean=np.mean(frame[...][...][0])
    Cb_mean=np.mean(frame[...][...][2])
    Cr_mean=np.mean(frame[...][...][1])

    #Rule Set - II
    '''
    def fu(val):
        return (-(2.6*pow(10,-10)*pow(val,7))
        +(3.3*pow(10,-7)*pow(val,6))
        -(1.7*pow(10,-4)*pow(val,5))
        +(5.16*pow(10,-2)*pow(val,4))
        -(9.10*pow(val,3))
        +(9.6*pow(10,2)*pow(val,2)
        -(5.6*pow(10,4)*val))
        +(1.4*pow(10,6)))

    def fl(val):
        return (-(6.77*pow(10,-8)*pow(val,5))
        +(5.5*pow(10,-5)*pow(val,4))
        -(1.76*pow(10,-2)*pow(val,3))
        +(2.78*pow(val,2))
        -(2.15*pow(10,2)*val)
        +(6.62*pow(10,3)))

    def fd(val):
        return ((1.81*pow(10,-4)*pow(val,4))
        -(1.02*pow(10,-1)*pow(val,3))
        +(2.17*10*pow(val,2))
        -(2.05*pow(10,3)*val)
        +(7.29*pow(10,4)))
    '''
    #iterate each pixel
    for x in range(frame_ycbcr.shape[1]):
        for y in range(frame_ycbcr.shape[0]):

            if frame_ycbcr[y][x][0]>frame_ycbcr[y][x][2] and frame_ycbcr[y][x][1]>frame_ycbcr[y][x][2]:
                if frame_ycbcr[y][x][0]>Y_mean and frame_ycbcr[y][x][2]<Cb_mean and frame_ycbcr[y][x][1]>Cr_mean:

            #if frame[y][x][2]>=fu(frame[y][x][1]) and frame[y][x][2]<=fd(frame[y][x][1]) and frame[y][x][2]>=fl(frame[y][x][1]):
            #if abs(frame_ycbcr[y][x][2]-frame_ycbcr[y][x][1])>200:
                    canvas[y][x][0]=225
                    canvas[y][x][1]=225
                    canvas[y][x][2]=225

                    pixel+=1

    return canvas,pixel

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

def Output(in_img,out_img):

    cv.imshow('CCTV 1 Stream Input',in_img)
    cv.imshow('CCTV 1 Stream Ouput',out_img)

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

        #iterating through each frame of the stream
        while(True):

            #extract frame
            frame=ExtractFrame(vid)
            
            #checking if the extracted frame is valid
            if str(type(frame))!="<class 'numpy.ndarray'>":
                break

            #localize fire
            in_img=frame.copy()
            start=time()
            out_img,pixel=Localize_Fire(frame)
            stop=time()

            #displaying output
            Output(in_img,out_img)

            #counting number of frames iterated
            count+=1

            #printing frame info
            os.system('clear')
            print("Frames Processed : "+str(count))
            print("FPS : "+str(round(1/(stop-start))))
            print("Pixels Detected : "+str(pixel))
 
            #waiting for quit keypress
            if cv.waitKey(1) & 0XFF == ord('q'):
                break

        #cleanup stream
        cv.destroyAllWindows()
        vid.release()

        #success
        print("Done Parsing Frames.")
