import cv2 as cv
import numpy as np

def VideoSetup(vidpath):
	vid=cv.VideoCapture(vidpath)
	return vid

def MaskSetup(first_frame):
	mask=np.zeros_like(first_frame)
	mask[..., 1]=255
	return mask

def VideoFrameExtract(vid):
	#Reading the input stream
	ret,rgb_frame=vid.read()
	#Conversion to grayscale -> Reduces computation as only the luminence channel is reuired
	grey_frame=cv.cvtColor(rgb_frame,cv.COLOR_BGR2GRAY)
	return ret,grey_frame,rgb_frame

def ImageOutput(input_frame,output_frame):
	cv.imshow("Input Video Stream",input_frame)
	cv.imshow("Dense Optical Flow",output_frame)
	
def DenseOpticalFlow(mask,prev_frame,next_frame):
	flow=cv.calcOpticalFlowFarneback(prev_frame,next_frame,None,0.5,3,15,3,5,1.1,0)
	mag,ang=cv.cartToPolar(flow[..., 0],flow[..., 1])
	mask[..., 0]=ang*180/np.pi/2
	mask[..., 2]=cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
	output=cv.cvtColor(mask,cv.COLOR_HSV2BGR)
	return output

if __name__=='__main__':
	#Setting up video file settings
	vidpath=input("Video File Path : ")
	vid=VideoSetup(vidpath)

	#Setting up HSV mask
	ret,first_frame_grey,first_frame_rgb=VideoFrameExtract(vid)
	if(ret==False):
		print("Unable to Read Video File !")
		exit(0)
	mask=MaskSetup(first_frame_rgb)
	prev_frame=first_frame_grey

	#Reading video stream
	while(vid.isOpened()):
		_,next_frame_grey,next_frame_rgb=VideoFrameExtract(vid)

		#Calculating Dense Optical Flow
		output=DenseOpticalFlow(mask,prev_frame,next_frame_grey)

		#Display Output
		ImageOutput(next_frame_rgb,output)

		#Awaiting exit key
		if cv.waitKey(1) & 0xFF==ord('q'):
			break

		#Swapping frames
		prev_frame=next_frame_grey

	#Cleaning up
	vid.release()
	cv.destroyAllWindows()