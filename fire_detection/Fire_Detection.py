import os
import cv2 as cv
import numpy as np
from time import time,sleep
from flicker_detection.Motion_Detection import Background_Subtraction,VideoSetup,Resize_Frame,Setup_Parameters
from color_detection.Color_Detection_YCbCr import Color_Detection,Setup_Plotter,ExtractFrame,Plot_Graph

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
    
    #Setting Up Video Stream Parameters
    vidpath=input("Video File Path : ")
    vid,status,fps=VideoSetup(vidpath)

    #Setting Downscale Factor(%)
    scale_percent=100-int(input("Enter Downscale Factor (in %) : "))

    plot_enable=input("Enable Real-Time Plotting ? [Y/N] : ")

    #Checking If Stream Is Open
    if(status==False):
        print("Error, Failed to Open Video Stream !")
        exit(0)

    else:
        
        #Setting Plotter Parameters
        if plot_enable=='Y' or plot_enable=='y':
            plt,X,Y=Setup_Plotter(vid,scale_percent)

        #Setting Up Image Processing Parameters
        kernel,bg_subtractor=Setup_Parameters(kern_size=10)

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
            out_img=frame.copy()

            #Start Timer For Calculating Real-Time FPS
            start=time()

            #Detect Fire-Colored Pixels
            color_mask,pixel=Color_Detection(frame)
            
            #Perform Closing Morphological Operation
            color_mask=cv.morphologyEx(color_mask,cv.MORPH_CLOSE,kernel)

            #Typecast to uint8
            color_mask=np.array(color_mask,np.uint8)

            #Remove Noise
            frame=cv.GaussianBlur(frame,(5,5),1)

            #Generate Foreground Mask
            fg_mask=Background_Subtraction(frame,bg_subtractor)

            #Stop Timer
            stop=time()

            #Initialize Empty Mask For Fusion
            mask=np.zeros_like(fg_mask)

            #Calculate Intersection of Foreground Mask and Color Mask
            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    if fg_mask[x][y]==color_mask[x][y]:
                        mask[x][y]=fg_mask[x][y]
                    else:
                        mask[x][y]=0

            #Checking If Generated Fused Mask Is Empty
            if mask.size!=0:

                #Perform Closing Morphological Operation
                mask=cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)

                #Region Boundary Estimation
                out_img,boxes=Region_Draw(mask,out_img)
            
            else:

                #If Fused Mask Is Empty -> Generate a Zero-Filled Mask
                mask=np.zeros_like(fg_mask)

            #Counting No. Of Frames Processed
            count+=1
            
            #Displaying Input, Mask and Final Output
            Output(in_img,fg_mask,color_mask,mask,out_img,"CCTV Input","Foreground Mask","Color Mask","Fused Mask","Output")

            #Plotting Graph
            if plot_enable=='Y' or plot_enable=='y':
                Plot_Graph(count,pixel,X,Y)

            #Printing Debug Info
            os.system('clear')
            print("Frames Processed : "+str(count))
            print("FPS : "+str(round(1/(stop-start))))
            print("Fire-Colored Pixels Detected : "+str(pixel))
            print("No. Of Bounding Boxes Generated : "+str(len(boxes)))
            print("Bounding Boxes : "+str(boxes))

 
            #Waiting For Keypress -> Quit OpenCV 'imshow()' Window
            if cv.waitKey(1) & 0XFF == ord('q'):
                break

        #Close Video Stream and Cleanup
        cv.destroyAllWindows()
        vid.release()

        #Print Success Message
        print("Done Parsing Frames.")