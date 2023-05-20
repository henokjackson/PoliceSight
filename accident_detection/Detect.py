import torch
import base64
import requests
import threading
import cv2 as cv
from numpy import random
from utils.plots import plot_one_box
from models.experimental import attempt_load
from utils.torch_utils import select_device,time_synchronized,TracedModel
from utils.general import check_img_size,non_max_suppression,scale_coords,strip_optimizer,set_logging

class opt:
    img_size=640
    conf_thres=0.25
    iou_thres=0.45
    device=''
    view_img=False
    save_txt=False
    save_conf=False
    nosave=False
    no_trace=True
    project=''
    name=''
    exist_ok=True
    augment=True
    agnostic_nms=True
    update=True
    classes=0
    weights='/home/henok/Documents/B.Tech/S7/Courses/CSD415 - Final Year Project/Code/Enhancement-of-Public-Safety-using-Computer-Vision-and-NLP/fire_detection/models/best.pt'
'''
def Video_Write(fps,dataset,size,sav_path):
    lock=threading.Lock()
    lock.acquire()
    vid_output=cv.VideoWriter("./output/"+str(sav_path)+".mp4",cv.VideoWriter_fourcc('m','p','4','v'),fps,size)
    for (_,frame) in dataset:
        vid_output.write(frame)
    print("Written To File !")
    lock.release()
'''
def Initialize():
    # Initialize
    set_logging()
    device = select_device(opt.device)

    # half precision only supported on CUDA
    half = device.type != 'cpu' 
    
    return device,half

def Load_YOLO(weights,device):
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    
    return model

def detect(dataset,model,half,device,fps,size,sav_path,video_buffer):

    frame_counter=0

    imgsz,trace=opt.img_size,not opt.no_trace

    #Load Model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for img,im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img,augment=opt.augment)[0]
        
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for det in pred:  # detections per image
            s,im0='',im0s
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                if n>0:
                    frame_counter=frame_counter+1

                #'''
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                #'''
            #'''
            #Print time (inference + NMS)
            file.write(str(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS\n'))
            #'''

    if (frame_counter/len(dataset))>=0.5:
        #Call VideoWriter Here
        #vid_write_thread=threading.Thread(target=Video_Write,name='Video Writer',args=(fps,dataset,size,sav_path))
        #vid_write_thread.start()
        #vid_write_thread.join()
        for frame in video_buffer:
            _,buffer=cv.imencode('.jpg',frame)
            frame_base64=base64.b64encode(buffer).decode()
            data={"incident_frame":frame_base64,"incident_type":"fire"}
            host='http://192.168.225.128:5001/update'
            response=requests.post(url=host,json=data)
            print(response)

def YOLODetect(dataset,weights,model,half,device,fps,size,sav_path,video_buffer):
    global file
    file=open("/home/henok/Documents/B.Tech/S7/Courses/CSD415 - Final Year Project/Code/Enhancement-of-Public-Safety-using-Computer-Vision-and-NLP/fire_detection/output/log.txt","a+")
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            if weights:
                detect(dataset,model,half,device,fps,size,sav_path,video_buffer)
                strip_optimizer(opt.weights)
        else:
            detect()