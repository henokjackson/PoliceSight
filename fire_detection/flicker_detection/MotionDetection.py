import os
from time import time
import cv2 as cv
import numpy as np

def Setup_Parameters(kern_size):
    
    #Setting Parameters For Morphologcal Processing
    #Generate Kernel Using Built-In Function -> Circular Kernel
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(kern_size,kern_size))

    #Setting Parameters For Background Subtraction Using KNN
    #Initialize Background Subtractor Object
    bg_subtractor=cv.createBackgroundSubtractorKNN(history=2, detectShadows=False)

    #Return The Kernel and Background Subtraction Object
    return kernel,bg_subtractor

def Background_Subtraction(frame,bg_subtractor):
   
    #Apply Background Subtraction To The Frame
    fg_mask=bg_subtractor.apply(frame)

    return fg_mask

def VideoSetup(vidpath):

    #Creating Video Stream Object
    vid=cv.VideoCapture(vidpath)

    #Checking Stream Status
    status=vid.isOpened()

    #Obtaining FPS
    fps=vid.get(cv.CAP_PROP_FPS)

    return vid,status,round(fps)

def ExtractFrame(vid):

    #Extracting Frame From Video Stream
    _,frame=vid.read()

    return frame

def Resize_Frame(frame,scale_percent):

    #Downscale The Frame
    frame=cv.resize(frame,(int(frame.shape[1]*(scale_percent/100)),int(frame.shape[0]*(scale_percent/100))),interpolation=cv.INTER_AREA)

    return frame

def Output(in_img,out_img,msg1,msg2):

    #Display Input
    cv.imshow(msg1,in_img)

    #Display Output
    cv.imshow(msg2,out_img)

if __name__=="__main__":

    #Initialize Frame Counter
    count=0
    
    #Setting Up Video Stream Parameters
    vidpath=input("Video File Path : ")
    vid,status,fps=VideoSetup(vidpath)

    #Setting Downscale Factor(%)
    scale_percent=100-int(input("Enter Downscale Factor (in %) : "))

    #Checking If Stream Is Open
    if(status==False):
        print("Error, Failed to Open Video Stream !")
        exit(0)
        
    else:

        #Setting Up Image Processing Parameters
        kernel,bg_subtractor=Setup_Parameters(kern_size=5)

        #Iterate Through Each Frame
        while(True):

            #Extract a Frame
            frame=ExtractFrame(vid)
            
            #Checking If Extracted Frame Is Valid
            if str(type(frame))!="<class 'numpy.ndarray'>":
                break
                
            #Downscaling Frame Size
            frame=Resize_Frame(frame,scale_percent)

            #############################
            # Extracting Moving Regions #
            #############################

            #Keep Original Frame Copy
            in_img=frame.copy()

            #Start Timer For Calculating Real-Time FPS
            start=time()
            
            #Remove Noise
            frame=cv.GaussianBlur(frame,(5,5),1)
            
            #Generate Foreground Mask
            mask=Background_Subtraction(frame,bg_subtractor)

            #Stop Timer
            stop=time()

            #Displaying Input, Masks, Fused Mask and Final Output
            Output(in_img,mask,"CCTV Live Input","Fire Detection Output Mask")

            #Counting No. Of Frames Processed
            count+=1

            #Printing Debug Info
            os.system('clear')
            print("Frames Processed : "+str(count))
            print("FPS : "+str(round(1/(stop-start))))
 
            #Waiting For Keypress -> Quit OpenCV 'imshow()' Window
            if cv.waitKey(1) & 0XFF == ord('q'):
                break

        #Close Video Stream and Cleanup
        cv.destroyAllWindows()
        vid.release()

        #Print Success Message
        print("Done Parsing Frames.")