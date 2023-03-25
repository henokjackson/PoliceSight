import os
from time import time
import cv2 as cv
import numpy as np

def Setup_Parameter():

    #Set a threshold for decision
    threshold=180

    #Define the matrix size for pixel frequency matrix
    matrix_size=16

    #Initialize the background subtractor
    bg_subtractor=cv.createBackgroundSubtractorMOG2(history=10, varThreshold=50, detectShadows=False)

    # Initialize a variable to store the previous frame
    prev_frame = None

    return prev_frame,bg_subtractor,matrix_size,threshold

def Flicker_Detect(curr_frame,prev_frame,bg_subtractor,matrix_size,threshold):
   
    # Apply background subtraction to the frame
    fg_mask=bg_subtractor.apply(curr_frame)

    # Convert the frame to grayscale
    gray=cv.cvtColor(curr_frame,cv.COLOR_BGR2GRAY)

    # Calculate the height and width of the image
    height,width=gray.shape

    # Calculate the size of each sub-matrix
    sub_size=int(height/matrix_size)

    # Initialize a list to store the fire detection results for each sub-matrix
    fire_detected = []

    # Loop through the sub-matrices
    for j in range(matrix_size):
        for k in range(matrix_size):

            # Extract the sub-matrix
            sub_matrix=gray[j*sub_size:(j+1)*sub_size,k*sub_size:(k+1)*sub_size]

            # Compare each pixel in the sub-matrix with the corresponding pixel in the previous frame
            if prev_frame is not None:
                prev_sub_matrix=prev_frame[j*sub_size:(j+1)*sub_size,k*sub_size:(k+1)*sub_size]
                pixel_diff=np.abs(sub_matrix.astype(np.int16)-prev_sub_matrix.astype(np.int16))
                pixel_diff[pixel_diff<threshold]=0
                pixel_diff[pixel_diff>=threshold]=1
                if pixel_diff.sum()>0 and fg_mask[j*sub_size:(j+1)*sub_size,k*sub_size:(k+1)*sub_size].sum()>0:
                    fire_detected.append(True)
                else:
                    fire_detected.append(False)
            else:
                # If this is the first frame, assume there is no fire
                fire_detected.append(False)

    # Reshape the fire detection results into a matrix
    fire_matrix=np.reshape(fire_detected,(matrix_size,matrix_size))

    fire=False

    # Check if fire was detected in the frame
    if True in fire_detected:
            print("Fire Detected !")
            fire=True
    return gray.copy(),cv.resize(fire_matrix.astype('uint8')*255,(512, 512),interpolation=cv.INTER_NEAREST),fire


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

    #Display input
    cv.imshow('CCTV 1 Stream Input',in_img)

    #Display input
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

        #setting up
        prev_frame,bg_subtractor,matrix_size,threshold=Setup_Parameter()

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
            prev_frame,out_img,fire=Flicker_Detect(frame,prev_frame,bg_subtractor,matrix_size,threshold)
            stop=time()

            #displaying output
            Output(in_img,out_img)

            #counting number of frames iterated
            count+=1

            #printing frame info
            os.system('clear')
            print("Frames Processed : "+str(count))
            print("FPS : "+str(round(1/(stop-start))))
            print("Fire : "+str(fire))
 
            #waiting for quit keypress
            if cv.waitKey(1) & 0XFF == ord('q'):
                break

        #cleanup stream
        cv.destroyAllWindows()
        vid.release()

        #success
        print("Done Parsing Frames.")