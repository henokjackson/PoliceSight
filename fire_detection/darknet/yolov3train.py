import os
import glob

os.system("sed -i 's/OPENCV=0/OPENCV=1/' Makefile")
os.system("sed -i 's/GPU=0/GPU=1/' Makefile")
os.system("sed -i 's/CUDNN=0/CUDNN=1/' Makefile")
os.system("make")

os.system("cp cfg/yolov3.cfg cfg/yolov3_training.cfg")

os.system("sed -i 's/batch=1/batch=64/' cfg/yolov3_training.cfg")
os.system("sed -i 's/subdivisions=1/subdivisions=16/' cfg/yolov3_training.cfg")
os.system("sed -i 's/max_batches = 500200/max_batches = 4000/' cfg/yolov3_training.cfg")
os.system("sed -i '610 s@classes=80@classes=2@' cfg/yolov3_training.cfg")
os.system("sed -i '696 s@classes=80@classes=2@' cfg/yolov3_training.cfg")
os.system("sed -i '783 s@classes=80@classes=2@' cfg/yolov3_training.cfg")
os.system("sed -i '603 s@filters=255@filters=21@' cfg/yolov3_training.cfg")
os.system("sed -i '689 s@filters=255@filters=21@' cfg/yolov3_training.cfg")
os.system("sed -i '776 s@filters=255@filters=21@' cfg/yolov3_training.cfg")


os.system("echo -e 'Fire\nNot Fire' > data/obj.names")

os.system("echo -e 'classes= 2\ntrain  = data/train.txt\nvalid  = data/test.txt\nnames = data/obj.names\nbackup = ../' > data/obj.data")

images_list = glob.glob("../dataset/train/images/*.jpg")
with open("data/train.txt", "w") as f:
    f.write("\n".join(images_list))

os.system("chmod +x ./darknet")
os.system("./darknet detector train data/obj.data cfg/yolov3_training.cfg darknet53.conv.74 -dont_show")
