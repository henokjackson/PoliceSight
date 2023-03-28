import os
import cv2 as cv
import numpy as np
from time import time,sleep
from matplotlib import pyplot as plt
def Localize_Fire(frame):

    #pixel count
    pixel=0
    
    #norm_val error count
    norm_error=0

    #convert image from BGR to YCbCr
    #frame_rgb=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    frame_ycbcr=cv.cvtColor(frame,cv.COLOR_BGR2YCR_CB)

    mask=np.zeros_like(frame_ycbcr)

    
    #spliting the channels
    Y,Cr,Cb=cv.split(frame_ycbcr)

    #'''
    #Since Y ranges from 16 to 235 rather than 16 to 240 (for cb and cr)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j]=int((224/219)*(Y[i][j]-16)+16)


    #Re-combine the normalized Y channel into the frame
    frame_ycbcr=cv.merge((Y,Cr,Cb))
    
    #'''
    #'''
    #Params for Rule set - 0
    Y_Cb_min=np.min(Y-Cb)
    Y_Cb_max=np.max(Y-Cb)
    Cr_Cb_min=np.min(Cr-Cb)
    Cr_Cb_max=np.max(Cr-Cb)

    '''
    #Override with constant parameters
    #min => min-max
    #max => max-min
    Y_Cb_min=16-240
    Y_Cb_max=235-16
    Cr_Cb_min=16-240
    Cr_Cb_max=240-16
    '''

    '''
    #Individual parameters
    Y_min=np.min(Y)
    Y_max=np.max(Y)
    Cb_min=np.min(Cb)
    Cb_max=np.max(Cb)
    Cr_min=np.min(Cr)
    Cr_max=np.max(Cr)
    '''


    '''
    #Rule Set - I
    Y_mean=np.mean(frame_ycbcr[...][...][0])
    Cb_mean=np.mean(frame_ycbcr[...][...][2])
    Cr_mean=np.mean(frame_ycbcr[...][...][1])
    '''
    

    '''
    #Rule Set - II
    def fu(val):
        return (-(2.6*pow(10,-10)*pow(val,7))
        +(3.3*pow(10,-7)*pow(val,6))
        -(1.7*pow(10,-4)*pow(val,5))
        +(5.16*pow(10,-2)*pow(val,4))
        -(9.10*pow(val,3))
        +(9.6*pow(10,2)*pow(val,2))
        -(5.6*pow(10,4)*val)
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
            '''
            if frame_ycbcr[y][x][0]>frame_ycbcr[y][x][2] and frame_ycbcr[y][x][1]>frame_ycbcr[y][x][2]:
                #if frame_ycbcr[y][x][0]>Y_mean and frame_ycbcr[y][x][2]<Cb_mean and frame_ycbcr[y][x][1]>Cr_mean:
                    
                #if (frame_ycbcr[y][x][2]>=fu(frame_ycbcr[y][x][1])) and (frame_ycbcr[y][x][2]<=fd(frame_ycbcr[y][x][1])) and (frame_ycbcr[y][x][2]<=fl(frame_ycbcr[y][x][1])):

            '''
            #Rule Set - 0
            '''
            norm_Y_Cb=2*(((frame_ycbcr[y][x][0]-frame_ycbcr[y][x][2])-(Y_min-Cb_max))/((Y_max-Cb_min)-(Y_min-Cb_max)))-1
            norm_Cr_Cb=2*(((frame_ycbcr[y][x][1]-frame_ycbcr[y][x][2])-(Cr_min-Cb_max))/((Cr_max-Cb_min)-(Cr_min-Cb_max)))-1
            '''

            Y_Cb=frame_ycbcr[y][x][0]-frame_ycbcr[y][x][2]
            Cr_Cb=frame_ycbcr[y][x][1]-frame_ycbcr[y][x][2]

            norm_Y_Cb=(2*((Y_Cb-Y_Cb_min)/(Y_Cb_max-Y_Cb_min)))-1
            norm_Cr_Cb=(2*((Cr_Cb-Cr_Cb_min)/(Cr_Cb_max-Cr_Cb_min)))-1

            if norm_Y_Cb>=-1 and norm_Y_Cb<=1 and norm_Cr_Cb>=-1 and norm_Cr_Cb<=1:
                #print("norm_Y_Cb : "+str(norm_Y_Cb))
                #print("norm_Cr_Cb : "+str(norm_Cr_Cb))
                #if(abs(norm_Y_Cb)>=0.1 and abs(norm_Y_Cb)<=0.7) and (abs(norm_Cr_Cb)>=0.5 and abs(norm_Cr_Cb)<=1):
                #if(abs(norm_Y_Cb)>=0.0 and abs(norm_Y_Cb)<=0.5):
                if norm_Cr_Cb>=-0.6 and norm_Cr_Cb<=0 and norm_Y_Cb>=0.2 and norm_Y_Cb<0.6:
            
                    mask[y][x][0]=225
                    mask[y][x][1]=225
                    mask[y][x][2]=225
                    pixel+=1
            #'''
            else:
                #print("norm_Cr_Cb : "+str(norm_Cr_Cb))
                #print("norm_Y_Cb : "+str(norm_Y_Cb))
                #print("Error.. Overflow !")
                norm_error+=1
                sleep(1000)
            #'''

    #mask=cv.cvtColor(mask,cv.COLOR_YCR_CB2RGB)
    
    mask=cv.cvtColor(cv.cvtColor(mask,cv.COLOR_YCR_CB2RGB),cv.COLOR_RGB2GRAY)
    _,mask=cv.threshold(mask,128,255,cv.THRESH_BINARY)

    contours,hierarchy=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame,contours,-1,(255,0,0),2,cv.LINE_4)
    
    for contour in contours:
        (x,y,w,h)=cv.boundingRect(contour)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    

    return mask,frame,pixel,norm_error

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

def Resize_Frame(frame,scale_percent):
    frame=cv.resize(frame,(int(frame.shape[1]*(scale_percent/100)),int(frame.shape[0]*(scale_percent/100))),interpolation=cv.INTER_AREA)
    return frame


def Output(in_img,out_img,mask):

    cv.imshow('CCTV 1 Stream Input',in_img)
    cv.imshow('CCTV 1 Stream Ouput',out_img)
    cv.imshow('CCTV 1 Stream Ouput Mask',mask)

if __name__=="__main__":

    #frame counter
    count=0

    #frame resize factor, default=10
    scale_percent=10
    
    #setting up video stream parameters
    vidpath=input("Video File Path : ")
    vid,status=VideoSetup(vidpath)

    #setup scale factor
    scale_percent=100-int(input("Enter Downscale Factor (in %) : "))

    #checking if video stream is open
    if(status==False):
        print("Error, Failed to Open Video Stream !")
        exit(0)
    else:

        #plotter parameters
        X=[]
        Y=[]
        plt.ylim([0,int(vid.get(cv.CAP_PROP_FRAME_WIDTH))*(scale_percent/100)*int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))*(scale_percent/100)])
        plt.xlabel("Time")
        plt.ylabel("Fire Pixels Detected")
        plt.title("YCbCr Fire Detection Model")

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
            mask,out_img,pixel,norm_error=Localize_Fire(frame)
            stop=time()

            #displaying output
            Output(in_img,out_img,mask)

            #counting number of frames iterated
            count+=1

            #updating plotter values
            X.append(count)
            Y.append(pixel)

            #plotting
            plt.plot(X,Y,color='red')
            plt.pause(0.00001)

            #printing frame info
            os.system('clear')
            print("Frames Processed : "+str(count))
            print("FPS : "+str(round(1/(stop-start))))
            print("Pixels Detected : "+str(pixel))
            print("Normalized Value Overflow : "+str(norm_error))
 
            #waiting for quit keypress
            if cv.waitKey(1) & 0XFF == ord('q'):
                break

        #cleanup video stream
        cv.destroyAllWindows()
        vid.release()

        #success
        print("Done Parsing Frames.")