import os
import cv2 as cv
import numpy as np
from time import time,sleep
from matplotlib import pyplot as plt

#Color Channel Thresholds - Global
norm_Y_Cb_LB=0.1
norm_Y_Cb_UB=0.4
norm_Cr_Cb_LB=0
norm_Cr_Cb_UB=0.3

def Setup_Parameters(kern_size):
    
    #Setting Parameters For Morphologcal Processing
    #Generate Kernel Using Built-In Function -> Circular Kernel
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(kern_size,kern_size))

    #Return The Kernel
    return kernel

def Color_Detection(frame):
    #Initialize Fire-Pixel Counter
    pixel=0
    
    #Initialize Normalization Error/Overflow Counter
    norm_error=0

    #Convert Frame From BGR to YCbCr
    frame_ycbcr=cv.cvtColor(frame,cv.COLOR_BGR2YCR_CB)

    #Initialize Zero-Filled Mask Of Frame Size
    mask=np.zeros(([frame_ycbcr.shape[0],frame_ycbcr.shape[1]]))

    #Splitting All Channels
    Y,Cr,Cb=cv.split(frame_ycbcr)


    #Calculating Min and Max Bounds Of Each Rule
    #Y_Cb_min=np.min(Y-Cb)
    #Y_Cb_max=np.max(Y-Cb)
    Cr_Cb_min=np.min(Cr-Cb)
    Cr_Cb_max=np.max(Cr-Cb)

    #Iterate Through Each Pixel
    for x in range(frame_ycbcr.shape[1]):
        for y in range(frame_ycbcr.shape[0]):

            #Check Basic Rule
            if frame_ycbcr[y][x][0]>frame_ycbcr[y][x][2] and frame_ycbcr[y][x][1]>frame_ycbcr[y][x][2]:
                #Y_Cb=frame_ycbcr[y][x][0]-frame_ycbcr[y][x][2]
                Cr_Cb=frame_ycbcr[y][x][1]-frame_ycbcr[y][x][2]

                #Perform Normalization To Range [-1,1]
                #norm_Y_Cb=(2*((Y_Cb-Y_Cb_min)/(Y_Cb_max-Y_Cb_min)))-1
                norm_Cr_Cb=(2*((Cr_Cb-Cr_Cb_min)/(Cr_Cb_max-Cr_Cb_min)))-1

                if abs(norm_Cr_Cb)>=norm_Cr_Cb_LB and abs(norm_Cr_Cb)<=norm_Cr_Cb_UB: #and abs(norm_Y_Cb)>=norm_Y_Cb_LB and abs(norm_Y_Cb)<=norm_Y_Cb_UB:
                    mask[y][x]=225
                    pixel+=1
                    
    #Binary Thresholding -  Channel Reduction
    _,mask=cv.threshold(mask,128,255,cv.THRESH_BINARY)
    return mask,pixel

def VideoSetup(vidpath):

    #Creating Video Stream Object
    vid=cv.VideoCapture(vidpath)

    #Checking Stream Status
    status=vid.isOpened()

    return vid,status


def ExtractFrame(vid):

    #Extracting Frame From Video Stream
    ret,frame=vid.read()

    return frame


def Resize_Frame(frame,scale_percent):
    
    #Downscale The Frame
    frame=cv.resize(frame,(int(frame.shape[1]*(scale_percent/100)),int(frame.shape[0]*(scale_percent/100))),interpolation=cv.INTER_AREA)
    
    return frame


def Mask_Process(kernel,mask):

    ####################################
    # Perform Morphological Processing #
    ####################################

    #Perform Closing Morphological Operation -> Form Blobs
    mask=cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)

    ##############################
    # Region Boundary Estimation #
    ##############################

    #Typecast to uint8
    mask=np.array(mask,np.uint8)

    #Binary Thresholding -  Channel Reduction
    _,mask=cv.threshold(mask,128,255,cv.THRESH_BINARY)

    return mask

def Setup_Plotter(vid,scale_percent):
    
    #Create Axes List
    X=[]
    Y=[]
    
    #Setting Axes Limits
    plt.ylim([0,int(vid.get(cv.CAP_PROP_FRAME_WIDTH))*(scale_percent/100)*int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))*(scale_percent/100)])
    
    #Setting Axes Labels
    plt.xlabel("Frames (Time)")
    plt.ylabel("Fire Pixels Detected")

    #Setting Plot Title
    plt.title("YCbCr Fire Detection Model")

    return plt,X,Y

def Plot_Graph(count,pixel,X,Y):

    #Updating Plot List Values 
    X.append(count)
    Y.append(pixel)

    #Plotting Graph
    plt.plot(X,Y,color='red')
    plt.pause(0.00001)

def Output(in_img,out_img,mask,msg1,msg2,msg3):

    #Display Input
    cv.imshow(msg1,in_img)

    #Display Output
    cv.imshow(msg2,out_img)

    #Display Output Mask
    cv.imshow(msg3,mask)


if __name__=='__main__':

    #Initialize Frame Counter
    count=0
    
    #Setting Up Video Stream Parameters
    vidpath=input("Video File Path : ")
    vid,status=VideoSetup(vidpath)

    #Setting Downscale Factor(%)
    scale_percent=100-int(input("Enter Downscale Factor (in %) : "))

    plot_enable=input("Enable Real-Time Plotting ? [Y/N] : ")

    #Checking If Stream Is Open
    if(status==False):
        print("Error, Failed to Open Video Stream !")
        exit(0)

    else:

        #Setting Up Image Processing Parameters
        kernel=Setup_Parameters(kern_size=5)

        #Setting Plotter Parameters
        if plot_enable=='Y' or plot_enable=='y':
            plt,X,Y=Setup_Plotter()

        #Iterate Through Each Frame
        while(True):

            #Extract a Frame
            frame=ExtractFrame(vid)
            
            #Checking If Extracted Frame Is Valid
            if str(type(frame))!="<class 'numpy.ndarray'>":
                break

            #Downscaling Frame Size
            frame=Resize_Frame(frame,scale_percent)

            ####################################
            # Extracting Fire-Coloured Regions #
            ####################################

            #Keep Original Frame Copy
            in_img=frame.copy()

            #Start Timer For Calculating Real-Time FPS
            start=time()

            #Detect Fire-Colored Pixels
            mask,pixel=Color_Detection(frame)

            #Stop Timer
            stop=time()

            #Perform Mask Processing
            mask,out_img=Mask_Process(kernel,mask,frame)

            #Displaying Input, Mask and Final Output
            Output(in_img,out_img,mask,"Input","Output","Color Mask")
            #Output(mask1,mask2,"Foreground Mask","Color Mask")

            #Counting No. Of Frames Processed
            count+=1
            
            #Plotting Graph
            if plot_enable=='Y' or plot_enable=='y':
                Plot_Graph(count,pixel,X,Y)

            #Printing Debug Info
            os.system('clear')
            print("Frames Processed : "+str(count))
            print("FPS : "+str(round(1/(stop-start))))
            print("Pixels Detected : "+str(pixel))
 
            #Waiting For Keypress -> Quit OpenCV 'imshow()' Window
            if cv.waitKey(1) & 0XFF == ord('q'):
                break

        #Close Video Stream and Cleanup
        cv.destroyAllWindows()
        vid.release()

        #Print Success Message
        print("Done Parsing Frames.")