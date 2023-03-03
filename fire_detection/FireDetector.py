import os
import cv2
import numpy as np

def Detect(img,net,classes):
    height,width, _=img.shape
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.75)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    return img

def VideoSetup(vidpath):

    net=cv2.dnn.readNet('darknet/model.weights', 'darknet/yolov7_testing.cfg')
    classes = []
    with open("darknet/classes.txt", "r") as f:
        classes = f.read().splitlines()

    #opening video stream
    vid=cv2.VideoCapture(vidpath)

    #checking stream status
    status=vid.isOpened()

    return vid,status,net,classes

def ExtractFrame(vid):
    #reading the video frame by frame
    ret,frame=vid.read()
    return frame

def Output(img):
    cv2.imshow('CCTV 1 Stream',img)

if __name__=="__main__":
    count=0
    vidpath=input("Video File Path : ")
    vid,status,net,classes=VideoSetup(vidpath)
    if(status==False):
        print("Unable to Open Video !")
        exit(0)
    else:
        while(True):
            frame=ExtractFrame(vid)
            if str(type(frame))!="<class 'numpy.ndarray'>":
                break
            img=Detect(frame,net,classes)
            Output(img)
            count+=1
            os.system('clear')
            print("Frames Processed : "+str(count))
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
        cv2.destroyAllWindows()
        vid.release()
        print("Done Parsing Frames.")
