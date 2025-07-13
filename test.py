import cv2
import time
from CameraDetect import CameraDetect
from Uart import Uart, Task
from LogisticsRobot import LogisticsRobot
import RPi.GPIO as GPIO

import cv2
import numpy as np
# import serial
import time
import struct
from collections import Counter
import threading
import subprocess
from typing import Dict, List, Tuple
# from detect import YOLOv5Detector  # 导入新的检测器类
import argparse
import torch

# cap_detect = cv2.VideoCapture("/dev/locate_video",cv2.CAP_V4L2)
# cap_detect.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) 

def main():
    uart = Uart()
    cam = CameraDetect(uart)
    cam._init_camera(1)  # 初始化摄像头
    fcn = LogisticsRobot(uart)
    # fcn.cam._init_camera(1) 
    angle = 0

    cap_detect = cv2.VideoCapture("/dev/detect_video",cv2.CAP_V4L2)
    cap_detect.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) 
    cap_detect.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_detect.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
    if not cap_detect.isOpened():
        print("摄像头打开失败！")
    else:
        print("摄像头已打开，分辨率:", 
            cap_detect.get(cv2.CAP_PROP_FRAME_WIDTH), 
            "x", 
            cap_detect.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:

        # ret, frame = cap_detect.read()
        # cv2.imshow("frame", frame)
        # ad = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        # aruco_params = cv2.aruco.DetectorParameters()
        # cv2.waitKey(1)

        # x, y = cam.locate_paper(1)
        # x, y = cam.locate_box()
        # x, y = cam.locate_paper_by_bottom_edge()
        # x, y = cam.locate_paper_by_edges()
        # x, y = cam.locate_paper_with_side_guides()
        x, y = cam.locate_paper_by_guided_edge(2)
        
        
        # x, y = cam.detectLine()
                






if __name__ == "__main__":
    main()