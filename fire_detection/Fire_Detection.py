import os
import threading
import cv2 as cv
import numpy as np
from time import time,sleep
from detect import YOLODetect,Load_YOLO,Initialize
from color_detection.Color_Detection_YCbCr import Color_Detection,Setup_Plotter,ExtractFrame,Plot_Graph
from flicker_detection.Motion_Detection import Background_Subtraction,VideoSetup,Resize_Frame,Setup_Parameters

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
weights='models/best.pt'

w1=2
w2=1
vid_length=3
detected_frames=0
frame_count_threshold=5

def YOLO(sav_path):
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
    if device!=None and half!=None and loaded==True:
        YOLODetect("./"+str(sav_path),weights,model,half,device)

def Image_Write(video_buffer,sav_path):
    os.mkdir("./"+str(sav_path))
    for i,frame in enumerate(video_buffer):
        cv.imwrite("./"+str(sav_path)+"/"+str(i)+".png",frame)

def Video_Write(fps,video_buffer,size):
    vid_output=cv.VideoWriter("./vids/"+str(sav_path)+".mp4",cv.VideoWriter_fourcc('m','p','4','v'),fps,size)
    for frame in video_buffer:
        vid_output.write(frame)
    print("Written To File !")

    yolo_detect_thread=threading.Thread(target=YOLO,name="YOLO Thread",args=("vids/",))
    yolo_detect_thread.start()


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

def Area_Randomness_Check(box1,box2):
    global detected_frames
    box1_area=box1[0][2]*box1[0][3]
    box2_area=box2[0][2]*box2[0][3]
    area_ratio=box2_area/box1_area
    print("\n#############################")
    print("# Box Area Randomness Check #")
    print("#############################")
    print("Previous Frame Area : "+str(box1_area))
    print("Current Frame Area : "+str(box2_area))
    print("Area Randomness Ratio : "+str(area_ratio))
    
    if area_ratio!=1 or area_ratio!=0:
        detected_frames=detected_frames+w1

def Box_Randomness_Check(box1,box2):
    global detected_frames
    count_ratio=box2/box1
    print("\n##############################")
    print("# Box Count Randomness Check #")
    print("##############################")
    print("Previous Frame Box Count : "+str(box1))
    print("Current Frame Box Count : "+str(box2))
    print("Box Count Randomness Ratio : "+str(count_ratio))
    
    if count_ratio!=1 or count_ratio!=0:
        detected_frames=detected_frames+w2

if __name__=='__main__':

    #Initialize Frame Counter
    count=0
    
    #Initialize Fire-Detected Frame Pointer
    pointer=-1

    #Initialize Save Video Buffer
    video_buffer=[]

    #Setting Video Save Path
    sav_path=0
    
    #Initialize Frame Queue
    frame_queue=[None,None]

    #Setting Up Video Stream Parameters
    vidpath=input("Video File Path : ")
    vid,status,fps=VideoSetup(vidpath)
    size=(int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)))

    #Calculate Video Buffer Size
    video_buffer_max_size=fps*vid_length

    #Setting Downscale Factor(%)
    scale_percent=100-int(input("Enter Downscale Factor (in %) : "))

    #Enable / Disable Real-Time Plotting
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

            #Keep Original Image Copy
            frame_org=frame.copy()

            #Downscaling Frame Size
            frame=Resize_Frame(frame,scale_percent)

            ####################################
            # Extracting Fire-Coloured Regions #
            ####################################

            #Keep Resized Frame Copy
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
            
            #Printing Debug Info
            #os.system('clear')
            print("Frames Processed : "+str(count))
            print("FPS : "+str(round(1/(stop-start))))
            print("Fire-Colored Pixels Detected : "+str(pixel))
            print("No. Of Bounding Boxes Generated : "+str(len(boxes)))
            print("Bounding Boxes : "+str(boxes))

            #First Two Frames Check
            if count==1:
                #Queueing Frame
                frame_queue[0]=(mask,pixel,boxes)
            elif count==2:
                #Queueing Frame
                frame_queue[1]=(mask,pixel,boxes)
            
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
                    #img_write_thread=threading.Thread(target=Image_Write,name='Image Writer',args=(video_buffer.copy(),sav_path))
                    vid_write_thread=threading.Thread(target=Video_Write,name='Video Writer',args=(fps,video_buffer.copy(),size))
                    
                    #Starting All Threads
                    #img_write_thread.start()
                    vid_write_thread.start()
                    
                    #Resetting Variables
                    detected_frames=0
                    video_buffer.clear()
                    sav_path=sav_path+1

            #Printing Debug Info
            print("Fire-Detected Frame Counts : "+str(detected_frames))
            print("Video File Length : "+str(len(video_buffer)/fps)+"s")
            print("Target File Length : "+str(vid_length)+"s")
            print("\n\n")

            #Displaying Input, Mask and Final Output
            Output(in_img,fg_mask,color_mask,mask,out_img,"CCTV Input","Foreground Mask","Color Mask","Fused Mask","Output")

            #Plotting Graph
            if plot_enable=='Y' or plot_enable=='y':
                Plot_Graph(count,pixel,X,Y)

            #Waiting For Keypress -> Quit OpenCV 'imshow()' Window
            if cv.waitKey(1) & 0XFF == ord('q'):
                break

        #Close Video Stream and Cleanup
        cv.destroyAllWindows()
        vid.release()

        #Print Success Message
        print("Done Parsing Frames.")