import math
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

dim_blue_min =  [90,25,84]#100 60 80
dim_blue_max =  [140,255,255]#124 230 255


class CameraDetect:
    def __init__(self,cam_mode = 0):
        """
        初始化Cameracontrol类的实例。

        参数:
            cam_mode (int): 摄像头类型，0为识别摄像头，1为定位摄像头。

       """
        self.frame=None
        # 识别结果存储
        self.detector = None  # YOLO检测器
        self.detection_results = {
            'shelf': {'upper': [], 'lower': []},
            'pallets': []  # 纸垛信息
        }
        # 初始化摄像头
        self._init_camera(cam_mode)
        self.zone_mapping = {k: None for k in ['a', 'b', 'c', 'd', 'e', 'f']}
        self.upper=[]
        self.lower=[]
        self.box_mapping={}
        self.cap_detect = None
        self.cap_locate = None

        # # 初始化YOLO检测器（只初始化一次）
        # if cam_mode == 0:
        #     self._init_detector()


    def _init_detector(self):
        """初始化YOLOv5检测器（只执行一次）"""
        weights = 'runs/train/exp3/weights/best.pt'
        data = 'data/papers.yaml'
        self.detector = YOLOv5Detector(
            weights=weights,
            data=data,
            imgsz=(640, 640),
            device='0' if torch.cuda.is_available() else 'cpu',  # 使用GPU加速
            half=True  # 使用半精度推理加速
        )
        print("YOLOv5 detector initialized")

    def _init_camera(self, cam_mode: int):
        if cam_mode == 0:  #识别摄像头
            try:
                self.cap_detect = cv2.VideoCapture("/dev/detect_video",cv2.CAP_V4L2) #V4l2树莓派
                if not self.cap_detect.isOpened():
                    raise ValueError("camera not open")
                self.cap_detect.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) 
                self.cap_detect.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap_detect.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
                for _ in range(5):
                    ret, _ = self.cap_detect.read()
                    time.sleep(0.01)

            except Exception as e:
                print(f"camera init error: {e}")
                raise
            
        elif(cam_mode ==1):  #定位摄像头

            self.cap_locate = cv2.VideoCapture("/dev/locate_video",cv2.CAP_V4L2) 
            # self.cap_locate = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
            print("initial")
            if not self.cap_locate.isOpened():
                raise ValueError("camera not open")
            self.cap_locate.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) 
            self.cap_locate.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap_locate.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

    def close_cam(self):
        self.cap.release()
    
    def close_windows(self):
        """
        释放摄像头资源，并关闭相关的图像显示窗口。
        """
        cv2.destroyAllWindows()
    
    def detect_zones(self, scan_phase):
        ret, frame = self.cap_detect.read()
        ret, frame = self.cap_detect.read()
        # --------------------- 扫描左侧区域 ---------------------
        h, w = frame.shape[:2]  # 获取图像尺寸
        # self.process_frame()# 处理当前帧
        # raw_detections = self.detector.raw_detections
        # 执行检测
        detections = self.process_frame(frame)
        if scan_phase == 0:
            zones = {
                'a': (0, w/3),      # 左1/3区域为a
                'b': (w/3, 2*w/3),  # 中间1/3为b
                'c': (2*w/3, w)     # 右1/3为c
            }
        elif scan_phase == 1:
            zones = {
                'd': (0, w/3),      # 左1/3区域为a
                'e': (w/3, 2*w/3),  # 中间1/3为b
                'f': (2*w/3, w)     # 右1/3为c
            }
        
        # 处理检测结果
        if detections is not None and len(detections) > 0:
            for det in detections:
                *xyxy, conf, cls_idx = det
                cls_name = self.detector.names[int(cls_idx)]
                if not cls_name.startswith('paper_'):
                    continue
                try:
                    current_num = int(cls_name.split('_')[1])
                except (IndexError, ValueError):
                    continue
                if current_num == 0:
                    continue
                x_center = (xyxy[0] + xyxy[2]) / 2
                # 寻找匹配区域
                for zone, (start, end) in zones.items():
                    if start <= x_center < end:
                        # 保留最接近区域中心的检测
                        zone_center = (start + end) / 2
                        current_value = self.zone_mapping[zone]
                        
                        if current_value is None or (abs(x_center - zone_center) < abs(current_value[0] - zone_center)):
                            self.zone_mapping[zone] = (current_num)
                        break


        if self.scan_phase == 1:
            print("最终映射：", self.zone_mapping)
            # 收集有效检测结果
            detected_numbers = [num for num in self.zone_mapping.values() if num is not None]
            # 数据校验
            if len(detected_numbers) != 5:
                raise ValueError(f"检测到{len(detected_numbers)}个纸垛，应为5个！")
            # 计算缺失数字
            missing_num = list(set(range(1,7)) - set(detected_numbers))[0]
            # 查找空位
            empty_positions = [pos for pos, num in self.zone_mapping.items() if num is None]
            if len(empty_positions) != 1:
                raise ValueError(f"发现{len(empty_positions)}个空位，应为1个！")
            print(f"分析结果：位置 {empty_positions[0]} 缺失编号 {missing_num}")

        return self.zone_mapping
    
    def detect_boxes(self):
        """
        根据Y坐标中值区分上下层，返回排序后的实际编号列表
        返回: (上层货箱编号列表, 下层货箱编号列表)
        """
        # 获取所有货箱检测结果
        ret, frame = self.cap_detect.read()
        ret, frame = self.cap_detect.read()
        detections = self.process_frame(frame)
        boxes = []
        
        if detections is not None and len(detections) > 0:
            for det in detections:
                *xyxy, conf, cls_idx = det
                cls_name = self.detector.names[int(cls_idx)]
                
                if not cls_name.startswith('box_'):
                    continue
                    
                try:
                    number = int(cls_name.split('_')[1])
                except (IndexError, ValueError):
                    continue
                
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2
                
                boxes.append({
                    'x': x_center,
                    'y': y_center,
                    'number': number
                })

        # 计算Y坐标中值
        y_values = [b['y'] for b in boxes]
        y_median = sorted(y_values)[len(y_values) // 2]

        # 分离上下层
        upper = [b for b in boxes if b['y'] < y_median]
        lower = [b for b in boxes if b['y'] >= y_median]

        # 按X坐标排序并提取编号
        upper_sorted = sorted(upper, key=lambda x: x['x'])
        lower_sorted = sorted(lower, key=lambda x: x['x'])

        # # 将结果存储到类的属性中
        # self.upper = [b['number'] for b in upper_sorted]
        # self.lower = [b['number'] for b in lower_sorted]

        return (
            [b['number'] for b in upper_sorted],
            [b['number'] for b in lower_sorted]
        )
        
    def process_frame(self, frame):
        """
        处理单帧并返回检测结果
        返回: 检测结果 tensor [x1, y1, x2, y2, conf, cls]
        """
        if self.detector is None:
            raise RuntimeError("Detector not initialized")
        
        return self.detector.detect(frame)

    def convert_box_layers_to_dict(self):
        """
        将车上上下两层货箱转换为字典表示，
        按照位置依次对应为：
        车上第二层 (上层) 从左到右: A, B, C
        车上第一层 (下层) 从左到右: D, E, F

        Returns:
            dict: 一个字典，键为位置代号 (A-F)，值为对应的货箱编号。
                如果货箱数量不符预期，则返回一个空字典或部分映射。
        """
        positions = ['A', 'B', 'C', 'D', 'E', 'F']
        layers = [self.upper, self.lower]

        for i, layer in enumerate(layers):
            if len(layer) != 3:
                print(f"警告：车上第{i + 1}层货箱数量不为3，无法进行准确映射。")
                continue
            self.box_mapping.update({positions[j]: layer[j] for j in range(3)})

        return self.box_mapping

    def locate_box(self):
        """
        法检测图像中的矩形物体定位
        :param debug: 是否输出调试信息和可视化结果
        :return: 包含矩形定位信息的字典列表
        """
        if not hasattr(self, 'frame') or self.cap_locate is None:
            print("未获取到有效图像帧")
            return []
        ret,self.frame=self.cap_locate.read()


        blue_min = np.array(dim_blue_min)
        blue_max = np.array(dim_blue_max)
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask3 = cv2.inRange(hsv, blue_min, blue_max)
        res = cv2.bitwise_and(self.frame, self.frame, mask=mask3)
        cv2.imshow("res", res)
        
        # 图像预处理
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)
    
 
        gamma = 0.5
        invgamma = 1 / gamma
        gamma_image = np.array(np.power((gray / 255.0), invgamma) * 255, dtype=np.uint8)
        cv2.imshow("gamma", gamma_image)
        # equalized = cv2.equalizeHist(gray)
        # # _, mask = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # # binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        # #                                 cv2.THRESH_BINARY, 11, 2)
        # _, binary_fixed = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        # blurred = cv2.GaussianBlur(res, (5, 5), 0)
        # _, bright = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        # kernel = np.ones((5, 5), np.uint8)
        # closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
        # opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        blured = cv2.blur(res, (5, 5))
        _, bright = cv2.threshold(blured, 10, 255, cv2.THRESH_BINARY)
        gray = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        closed1 = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        # closed = cv2.morphologyEx(closed1, cv2.MORPH_CLOSE, kernel)
        # 边缘检测
        edges = cv2.Canny(opened, 50, 150)
        # cv2.imshow("gray", gray)
        # cv2.imshow("equalized", equalized)
        # cv2.imshow("bright", bright)
        # cv2.imshow("mask", mask3)
        # cv2.imshow("opened", opened)
        # cv2.imshow("edges", edges)
        cv2.imshow("closed1", closed1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(closed1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 获取图像尺寸
        h, w = self.frame.shape[:2]
        src=self.frame.copy()
        
        # locations = []
        x_offset = 0
        y_offset = 0
        for contour in contours:
            # 轮廓近似（简化多边形）
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
            # 计算轮廓面积
            area = cv2.contourArea(approx)
            if area > 10000:  # 最小面积阈值
                print("area:",area)
                # 获取边界框
                x, y, w_rect, h_rect = cv2.boundingRect(approx)
                
                # 计算中心点（左上角为原点）
                x_center = x + w_rect / 2.0
                y_center = y + h_rect / 2.0
                
                # 转换为以图像中心为原点的坐标
                x_offset = x_center - w/2
                y_offset = h/2 - y_center  # Y轴取反（向上为正）
                
                    # 在图像上绘制结果
                cv2.drawContours(self.frame, [approx], -1, (0, 255, 0), 2)
                cv2.circle(self.frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
                
                # 绘制边界框
                cv2.rectangle(self.frame, (x, y), (x + w_rect, y + h_rect), (255, 0, 0), 2)
                
                # # 显示原始中心坐标
                # cv2.putText(self.frame, f"({int(x_center)}, {int(y_center)})", 
                #         (int(x_center) + 10, int(y_center)), 
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # 显示相对于中心的坐标
                cv2.putText(self.frame, f"Center: ({x_offset:.1f}, {y_offset:.1f})", 
                        (int(x_center) + 10, int(y_center) + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
                
                # print(f"检测到矩形物体:")
                # print(f"原始中心坐标: ({x_center:.1f}, {y_center:.1f})")
                # print(f"相对中心坐标: ({x_offset:.1f}, {y_offset:.1f})")
                # print(f"边界框: x={x}, y={y}, 宽={w_rect}, 高={h_rect}")
                # print(f"面积: {area:.1f} 像素\n")

            
        
        # if debug and locations:
        cv2.imshow("Rectangle Detection", self.frame)
        cv2.waitKey(1)

        return x_offset, y_offset


    def locate_box1(self):
        if not hasattr(self, 'frame') or self.cap_locate is None:
            print("未获取到有效图像帧")
            return 0, 0  # 返回默认值
        
        ret, self.frame = self.cap_locate.read()
        if not ret:
            return 0, 0
        
        # 1. 改进的颜色处理 - 增加自适应阈值
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        
        # 使用多个颜色范围提高鲁棒性
        mask1 = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))  # 标准蓝色范围
        mask2 = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([110, 255, 255]))   # 稍宽的范围
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 2. 改进的形态学操作 - 更精细的噪声处理
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # 3. 轮廓检测改进 - 使用更稳定的方法
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果没有检测到轮廓，返回默认值
        if not contours:
            cv2.imshow("Result", self.frame)
            cv2.waitKey(1)
            return 0, 0
        
        # 4. 选择最大轮廓（假设只有一个主要目标）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 5. 使用最小外接矩形提高稳定性
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 计算中心点
        center = rect[0]
        x_center, y_center = center
        
        # 6. 卡尔曼滤波减少跳动（可选）
        if not hasattr(self, 'kalman_filter'):
            # 初始化卡尔曼滤波器
            self.kalman_filter = cv2.KalmanFilter(4, 2)
            self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            self.kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
            self.last_measurement = np.array([[np.float32(x_center)], [np.float32(y_center)]])
            self.last_prediction = self.last_measurement
        
        # 预测
        prediction = self.kalman_filter.predict()
        
        # 更新测量值
        measurement = np.array([[np.float32(x_center)], [np.float32(y_center)]])
        
        # 如果移动距离过大，可能是噪声，使用预测值
        dist = np.linalg.norm(measurement - self.last_measurement)
        if dist > 50:  # 阈值根据实际情况调整
            x_center, y_center = prediction[0], prediction[1]
        else:
            # 更新卡尔曼滤波器
            self.kalman_filter.correct(measurement)
            x_center, y_center = prediction[0], prediction[1]
            self.last_measurement = measurement
        
        # 获取图像尺寸
        h, w = self.frame.shape[:2]
        
        # 转换为以图像中心为原点的坐标
        x_offset = float(x_center - w/2)
        y_offset = float(h/2 - y_center)  # Y轴取反（向上为正）
        
        # 7. 在图像上绘制结果
        cv2.drawContours(self.frame, [box], 0, (0, 255, 0), 2)
        cv2.circle(self.frame, (int(x_center), int(y_center)), 10, (0, 0, 255), -1)
        
        # 显示坐标信息
        cv2.putText(self.frame, f"Offset: ({x_offset:.1f}, {y_offset:.1f})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Result", self.frame)
        cv2.waitKey(1)

        return x_offset, y_offset
    
    def locate_paper1(self):
        """
        法检测图像中的矩形物体定位
        :param debug: 是否输出调试信息和可视化结果
        :return: 包含矩形定位信息的字典列表
        """
        if not hasattr(self, 'frame') or self.cap_locate is None:
            print("未获取到有效图像帧")
            return []
        ret,self.frame=self.cap_locate.read()



        
        # 图像预处理
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        equalized = cv2.equalizeHist(gray)
        # _, mask = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                                 cv2.THRESH_BINARY, 11, 2)
        _, binary_fixed = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        # _, bright = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # 边缘检测
        edges = cv2.Canny(opened, 50, 150)
        # cv2.imshow("gray", gray)
        # cv2.imshow("equalized", equalized)
        # cv2.imshow("bright", bright)
        # cv2.imshow("mask", mask3)
        cv2.imshow("opened", opened)
        cv2.imshow("edges", edges)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 获取图像尺寸
        h, w = self.frame.shape[:2]
        src=self.frame.copy()
    
        
        # for contour in contours:
        #     # 轮廓近似（简化多边形）
        #     peri = cv2.arcLength(contour, True)
        #     approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
        #     # 计算轮廓面积
        #     area = cv2.contourArea(approx)
        #     if area > 5000:  # 最小面积阈值
                
        #         # 获取边界框
        #         x, y, w_rect, h_rect = cv2.boundingRect(approx)
                
        #         # 计算中心点（左上角为原点）
        #         x_center = x + w_rect / 2.0
        #         y_center = y + h_rect / 2.0
                
        #         # 转换为以图像中心为原点的坐标
        #         x_offset = x_center - w/2
        #         y_offset = h/2 - y_center  # Y轴取反（向上为正）
                
        #             # 在图像上绘制结果
        #         cv2.drawContours(self.frame, [approx], -1, (0, 255, 0), 2)
        #         cv2.circle(self.frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
                
        #         # 绘制边界框
        #         cv2.rectangle(self.frame, (x, y), (x + w_rect, y + h_rect), (255, 0, 0), 2)
                
        #         # 显示原始中心坐标
        #         cv2.putText(self.frame, f"({int(x_center)}, {int(y_center)})", 
        #                 (int(x_center) + 10, int(y_center)), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
        #         # 显示相对于中心的坐标
        #         cv2.putText(self.frame, f"Center: ({x_offset:.1f}, {y_offset:.1f})", 
        #                 (int(x_center) + 10, int(y_center) + 20), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
                
        #         # print(f"检测到矩形物体:")
        #         # print(f"原始中心坐标: ({x_center:.1f}, {y_center:.1f})")
        #         # print(f"边界框: x={x}, y={y}, 宽={w_rect}, 高={h_rect}")
        #         # print(f"面积: {area:.1f} 像素\n")

        x_offset = 0
        y_offset = 0

        if len(contours) > 0:
            # 找出面积最大的轮廓
            max_contour = max(contours, key=lambda c: cv2.contourArea(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)))
            
            # 对最大轮廓进行多边形近似
            peri = cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)
            
            # 计算轮廓面积
            area = cv2.contourArea(approx)
            if area > 5000:  # 最小面积阈值
                # 获取边界框
                x, y, w_rect, h_rect = cv2.boundingRect(approx)
                
                # 计算中心点（左上角为原点）
                x_center = x + w_rect / 2.0
                y_center = y + h_rect / 2.0
                
                # 转换为以图像左下角为原点的坐标
                x_offset = x_center
                y_offset = w - y_center  # Y轴取反（向上为正）
                
                # 在图像上绘制结果
                cv2.drawContours(self.frame, [approx], -1, (0, 255, 0), 2)
                cv2.circle(self.frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
                
                # 绘制边界框
                cv2.rectangle(self.frame, (x, y), (x + w_rect, y + h_rect), (255, 0, 0), 2)
                
                # # 显示原始中心坐标
                # cv2.putText(self.frame, f"({int(x_center)}, {int(y_center)})", 
                #         (int(x_center) + 10, int(y_center)), 
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # 显示相对于中心的坐标
                cv2.putText(self.frame, f"Center: ({x_offset:.1f}, {y_offset:.1f})", 
                        (int(x_center) + 10, int(y_center) + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


    
        cv2.imshow("Rectangle Detection", self.frame)
        cv2.waitKey(1)

        return x_offset, y_offset
    
    def locate_paper(self, rect_count=1):
        """
        检测图像中的矩形物体定位
        :param rect_count: 1=单个矩形，2=只识别中线左侧矩形，3=只识别中线右侧矩形
        :return: 包含矩形定位信息的元组 (x_offset, y_offset)
        """
        if not hasattr(self, 'frame') or self.cap_locate is None:
            print("未获取到有效图像帧")
            return 0, 0
        ret, self.frame = self.cap_locate.read()
        src = self.frame.copy()
        
        if not ret:
            print("读取图像失败")
            return 0, 0

        # 图像预处理
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        gamma = 0.5
        invgamma = 1 / gamma
        gamma_image = np.array(np.power((gray / 255.0), invgamma) * 255, dtype=np.uint8)
        cv2.imshow("gamma", gamma_image)
        # ret, binary = cv2.threshold(gamma_image,100,250,cv2.THRESH_BINARY)
        # cv2.imshow("erzhi",binary)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # equalized = clahe.apply(gray)
        # 高斯模糊
        blurred1 = cv2.GaussianBlur(gamma_image, (9, 9), 2)
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(blurred1, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("closed",closed)
        # cv2.imshow("opened",opened)

        # # 边缘检测
        edges = cv2.Canny(closed, 50, 150)
        kernel1 = np.ones((5, 5), np.uint8)
        clodes_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel1, iterations=2)
        _, binary_edges = cv2.threshold(clodes_edges, 128, 255, cv2.THRESH_BINARY)
        cv2.imshow("edges", edges)

        
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 获取图像尺寸
        h, w = self.frame.shape[:2]
        midline_x = w // 2  # 计算视野中线位置
        src = self.frame.copy()
        
        # 存储所有符合条件的矩形信息
        valid_rects = []
        
        # 收集所有符合条件的矩形
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
            area = cv2.contourArea(approx)
            
            
            # 面积阈值过滤
            if area > 5000:
                # print("area:",area)
                x, y, w_rect, h_rect = cv2.boundingRect(approx)
                # if (w_rect / h_rect) < 1.5 and (w_rect / h_rect) > 1.3:
                x_center = x + w_rect / 2.0
                y_center = y + h_rect / 2.0
                
                # 存储矩形信息 (中心坐标, 轮廓, 边界框, 面积)
                valid_rects.append({
                    'center': (x_center, y_center),
                    'contour': approx,
                    'bbox': (x, y, w_rect, h_rect),
                    'area': area
                })
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)       # 获取四个顶点坐标
                box = np.int0(box)             # 转换为整数（OpenCV坐标要求）
                
                # 绘制旋转矩形（绿色，线宽2）
                cv2.drawContours(self.frame, [box], 0, (0, 0, 255), 3)
        
        # 根据rect_count参数选择矩形
        selected_rect = None
        
        if rect_count == 1:  # 选择面积最大的矩形
            if valid_rects:
                selected_rect = max(valid_rects, key=lambda r: r['area'])
        
        elif rect_count == 2:  # 只选择中线左侧的矩形
            left_rects = [r for r in valid_rects if r['center'][0] < midline_x]
            if left_rects:
                selected_rect = max(left_rects, key=lambda r: r['area'])
        
        elif rect_count == 3:  # 只选择中线右侧的矩形
            right_rects = [r for r in valid_rects if r['center'][0] > midline_x]
            if right_rects:
                selected_rect = max(right_rects, key=lambda r: r['area'])
        
        # 处理选中的矩形
        if selected_rect:
            x_center, y_center = selected_rect['center']
            approx = selected_rect['contour']
            x, y, w_rect, h_rect = selected_rect['bbox']
            
            # 计算以图像左下角为原点的坐标
            x_offset = x_center
            y_offset = h - y_center  # 转换为左下角原点，Y轴向上为正
            
            # 绘制选中的矩形
            cv2.drawContours(self.frame, [approx], -1, (0, 255, 0), 2)
            cv2.circle(self.frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
            cv2.rectangle(self.frame, (x, y), (x + w_rect, y + h_rect), (255, 0, 0), 2)
            
            # 显示坐标信息
            cv2.putText(self.frame, f"Center: ({x_offset:.1f}, {y_offset:.1f})", 
                    (int(x_center) + 10, int(y_center) + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 绘制中线参考线
            cv2.line(self.frame, (midline_x, 0), (midline_x, h), (0, 255, 255), 1)
            cv2.putText(self.frame, f"Midline", 
                    (midline_x + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            x_offset, y_offset = 0, 0
            # 绘制中线参考线
            cv2.line(self.frame, (midline_x, 0), (midline_x, h), (0, 255, 255), 1)
            cv2.putText(self.frame, f"Midline", 
                    (midline_x + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 显示处理结果
        cv2.imshow("Rectangle Detection", self.frame)
        cv2.waitKey(1)

        return x_offset, y_offset


    def locate_paper_by_guided_edge111(self, rect_count=1, paper_width_range=(150, 250), paper_height_pixels=145):
        """
        通过侧边线引导，几何重构底边线，精确定位纸垛。
        支持根据位置和大小选择特定纸垛。
        :param rect_count: 1=视野中最大, 2=中线左侧最大, 3=中线右侧最大
        :param paper_width_range: 纸垛在图像中的像素宽度范围 (min_width, max_width)
        :param paper_height_pixels: 纸垛的估计像素高度
        :return: (x_offset, y_offset)
        """
        if self.cap_locate is None:
            print("相机未初始化")
            return 0, 0
            
        ret, self.frame = self.cap_locate.read()
        if not ret or self.frame is None:
            print("读取图像失败")
            return 0, 0

        src_for_display = self.frame.copy()
        h, w = self.frame.shape[:2]
        midline_x = w // 2

        # --- 步骤 1: 预处理和霍夫变换 ---
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        gamma = 0.5
        invgamma = 1 / gamma
        gamma_image = np.array(np.power((gray / 255.0), invgamma) * 255, dtype=np.uint8)
        cv2.imshow("gamma", gamma_image)
        # 高斯模糊
        blurred1 = cv2.GaussianBlur(gamma_image, (9, 9), 2)
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(blurred1, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.Canny(closed, 50, 150)
        kernel1 = np.ones((5, 5), np.uint8)
        clodes_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel1, iterations=2)
        _, binary_edges = cv2.threshold(clodes_edges, 128, 255, cv2.THRESH_BINARY)
        binary_edges[350:,:] = 0
        cv2.imshow("Edges for Line Detection", binary_edges)

        
        lines = cv2.HoughLinesP(binary_edges, 1, np.pi / 180, threshold=30, minLineLength=100, maxLineGap=25)
        
        if lines is None:
            # 即使没找到线，也要显示中线并返回
            cv2.line(src_for_display, (midline_x, 0), (midline_x, h), (0, 255, 255), 1)
            cv2.imshow("Paper Detection", src_for_display)
            cv2.waitKey(1)
            return 0, 0

        # --- 步骤 2: 找到所有可能的纸垛候选者 ---
        horizontal_lines = []
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle < 10 or abs(angle - 180) < 10:
                horizontal_lines.append({'line': line[0], 'y_pos': (y1 + y2) / 2})
            elif abs(angle - 90) < 10:
                vertical_lines.append({'line': line[0], 'x_pos': (x1 + x2) / 2})
        
        paper_candidates = []
        # 遍历所有垂直线组合，寻找侧边线对
        for i in range(len(vertical_lines)):
            for j in range(i + 1, len(vertical_lines)):
                line1_info = vertical_lines[i]
                line2_info = vertical_lines[j]
                
                pair_width = abs(line1_info['x_pos'] - line2_info['x_pos'])
                if not (paper_width_range[0] < pair_width < paper_width_range[1]):
                    continue
                
                left_edge, right_edge = (line1_info, line2_info) if line1_info['x_pos'] < line2_info['x_pos'] else (line2_info, line1_info)
                x_left, x_right = left_edge['x_pos'], right_edge['x_pos']
                
                # 在这对侧边线之间，寻找最靠下的那条水平线
                candidate_bottom_lines = []
                for h_line_info in horizontal_lines:
                    # 确保水平线在侧边线之间，并且在图像下半区
                    if x_left < (h_line_info['line'][0] + h_line_info['line'][2]) / 2 < x_right and h_line_info['y_pos'] > h * 0.4:
                        candidate_bottom_lines.append(h_line_info)
                
                if not candidate_bottom_lines:
                    continue

                # 选择最靠下的那条水平线作为底边基准 (y坐标最大)
                best_bottom_line_info = max(candidate_bottom_lines, key=lambda item: item['y_pos'])
                y_bottom = best_bottom_line_info['y_pos']
                
                # --- 几何重构 & 计算中心点 ---
                # 用推算出的坐标计算中心
                center_offset = paper_height_pixels / 2
                x_center = (x_left + x_right) / 2
                y_center = y_bottom - center_offset
                
                # 存储候选者信息，包括用于绘制的几何信息和用于选择的中心点/宽度
                paper_candidates.append({
                    'center': (x_center, y_center),
                    'width': pair_width,
                    'geo': {'x_left': x_left, 'x_right': x_right, 'y_bottom': y_bottom}
                })

        # --- 步骤 3: 根据 rect_count 选择最终的纸垛 ---
        selected_paper = None
        if not paper_candidates:
            print("未找到符合条件的纸垛候选者")
        elif rect_count == 1:
            selected_paper = max(paper_candidates, key=lambda p: p['width'])
        elif rect_count == 2:
            left_papers = [p for p in paper_candidates if p['center'][0] < midline_x]
            if left_papers:
                selected_paper = max(left_papers, key=lambda p: p['width'])
        elif rect_count == 3:
            right_papers = [p for p in paper_candidates if p['center'][0] > midline_x]
            if right_papers:
                selected_paper = max(right_papers, key=lambda p: p['width'])

        # --- 步骤 4: 处理和显示结果 ---
        x_offset, y_offset = 0, 0
        if selected_paper:
            geo = selected_paper['geo']
            x_left, x_right, y_bottom = int(geo['x_left']), int(geo['x_right']), int(geo['y_bottom'])
            
            x_center, y_center = selected_paper['center']
            x_offset = x_center
            y_offset = h - y_center
            
            # 绘制重构的、完整的底边 (粗绿色)
            cv2.line(src_for_display, (x_left, y_bottom), (x_right, y_bottom), (0, 255, 0), 4)
            
            # 推算并绘制左右边 (只绘制下半部分，更美观)
            y_top_half = y_bottom - int(paper_height_pixels / 2)
            y_top = y_bottom - int(paper_height_pixels)
            cv2.line(src_for_display, (x_left, y_bottom), (x_left, y_top), (255, 0, 0), 2)
            cv2.line(src_for_display, (x_right, y_bottom), (x_right, y_top), (255, 0, 0), 2)
            
            # 绘制估算的中心点和坐标
            cv2.circle(src_for_display, (int(x_center), int(y_center)), 7, (0, 0, 255), -1)
            cv2.putText(src_for_display, f"XY: ({x_offset:.1f}, {y_offset:.1f})", 
                        (int(x_center) + 10, int(y_center)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        # 总是绘制中线
        cv2.line(src_for_display, (midline_x, 0), (midline_x, h), (0, 255, 255), 1)
        cv2.putText(src_for_display, f"Midline", 
                    (midline_x + 5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("Paper Detection", src_for_display)
        cv2.waitKey(1)

        return x_offset, y_offset





    # 新增辅助函数：计算直线方程 y = mx + c 或 x = c
    def get_line_equation(self, line):
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:  # 垂直线
            return (np.inf, x1) # 斜率无穷大，截距为x坐标
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        return (m, c)

    # 新增辅助函数：计算两条直线方程的交点
    def get_intersection(self, eq1, eq2):
        m1, c1 = eq1
        m2, c2 = eq2

        if m1 == m2: # 平行线，无交点
            return None
        
        if m1 == np.inf: # 第一条是垂直线
            x = c1
            y = m2 * x + c2
            return int(x), int(y)
        
        if m2 == np.inf: # 第二条是垂直线
            x = c2
            y = m1 * x + c1
            return int(x), int(y)

        # 普通情况
        x = (c2 - c1) / (m1 - m2)
        y = m1 * x + c1
        return int(x), int(y)



    def locate_paper_by_guided_edge(self, rect_count=1, paper_width_range=(100, 300), paper_height_pixels=145):
        # ... (你的初始化和图像预处理部分保持不变) ...
        # 直到 lines = cv2.HoughLinesP(...)
        if self.cap_locate is None:
            print("相机未初始化")
            return 0, 0
            
        ret, self.frame = self.cap_locate.read()
        if not ret or self.frame is None:
            print("读取图像失败")
            return 0, 0

        src_for_display = self.frame.copy()
        h, w = self.frame.shape[:2]
        midline_x = w // 2

        # --- 步骤 1: 预处理和霍夫变换 ---
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        gamma = 0.5
        invgamma = 1 / gamma
        gamma_image = np.array(np.power((gray / 255.0), invgamma) * 255, dtype=np.uint8)
        cv2.imshow("gamma", gamma_image)
        # 高斯模糊
        blurred1 = cv2.GaussianBlur(gamma_image, (9, 9), 2)
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(blurred1, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.Canny(closed, 50, 150)
        kernel1 = np.ones((5, 5), np.uint8)
        clodes_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel1, iterations=2)
        _, binary_edges = cv2.threshold(clodes_edges, 128, 255, cv2.THRESH_BINARY)
        binary_edges[350:,:] = 0
        cv2.imshow("Edges for Line Detection", binary_edges)

        
        lines = cv2.HoughLinesP(binary_edges, 1, np.pi / 180, threshold=25, minLineLength=100, maxLineGap=50)
        
        if lines is None:
            # ... (处理未找到线的情况) ...
            print("no lines")
            cv2.line(src_for_display, (midline_x, 0), (midline_x, h), (0, 255, 255), 1)
            cv2.imshow("Paper Detection", src_for_display)
            cv2.waitKey(1)
            return 0, 0

        # --- 步骤 2: 找到所有可能的纸垛候选者 (修改版) ---
        # (这部分选择逻辑保持不变，因为我们需要找到最好的三条线)
        all_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            all_lines.append({
                'line': line[0],
                'angle': angle,
                'length': length,
                'center': ((x1+x2)/2, (y1+y2)/2)
            })
        
        paper_candidates = []
        side_lines = [l for l in all_lines if abs(abs(l['angle']) - 90) < 20]
        bottom_lines = [l for l in all_lines if abs(l['angle']) < 20 or abs(l['angle']-180) < 20]

        for i in range(len(side_lines)):
            for j in range(i + 1, len(side_lines)):
                left_edge = side_lines[i]
                right_edge = side_lines[j]
                if abs(left_edge['angle'] - right_edge['angle']) > 10: continue
                if left_edge['center'][0] > right_edge['center'][0]: left_edge, right_edge = right_edge, left_edge
                pair_width = abs(left_edge['center'][0] - right_edge['center'][0])
                if not (paper_width_range[0] < pair_width < paper_width_range[1]): continue
                
                side_center_y_avg = (left_edge['center'][1] + right_edge['center'][1]) / 2
                candidate_bottoms = []
                for bottom_edge in bottom_lines:
                    if not (left_edge['center'][0] < bottom_edge['center'][0] < right_edge['center'][0]): 
                        continue
                    angle_diff = abs(abs(bottom_edge['angle']) - abs(left_edge['angle']))
                    if not (80 < angle_diff < 100):
                        continue
                    h_edge_center_y = bottom_edge['center'][1]
                    print("bottom",h_edge_center_y,"side_average",side_center_y_avg )
                    # 底边的中点Y坐标必须大于(更靠下)侧边中点的平均Y坐标
                    if h_edge_center_y < side_center_y_avg:
                        continue
                    candidate_bottoms.append(bottom_edge)

                if not candidate_bottoms: continue
                best_bottom_edge = max(candidate_bottoms, key=lambda b: b['center'][1])
                
                # --- 计算中心点 (不再推算顶边) ---
                # 我们仍然用交点来获得精确的底边位置
                eq_left = self.get_line_equation(left_edge['line'])
                eq_right = self.get_line_equation(right_edge['line'])
                eq_bottom = self.get_line_equation(best_bottom_edge['line'])
                p_bottom_left = self.get_intersection(eq_left, eq_bottom)
                p_bottom_right = self.get_intersection(eq_right, eq_bottom)
                if p_bottom_left is None or p_bottom_right is None: continue

                # 中心点计算：底边中点向上偏移
                bottom_mid_x = (p_bottom_left[0] + p_bottom_right[0]) / 2
                bottom_mid_y = (p_bottom_left[1] + p_bottom_right[1]) / 2

                # “向上”的方向向量，垂直于底边
                bottom_angle_rad = math.radians(best_bottom_edge['angle'])
                # Y轴向下，所以向上是角度-90度
                up_angle_rad = bottom_angle_rad - math.pi / 2 
                
                offset = paper_height_pixels / 2
                dx = offset * math.cos(up_angle_rad)
                dy = offset * math.sin(up_angle_rad)

                x_center = bottom_mid_x + dx
                y_center = bottom_mid_y + dy

                # 存储候选者信息，注意现在'geo'存储的是三条原始线段
                paper_candidates.append({
                    'center': (x_center, y_center),
                    'width': pair_width,
                    'geo': {
                        'left_line': left_edge['line'],
                        'right_line': right_edge['line'],
                        'bottom_line': best_bottom_edge['line']
                    }
                })

        # --- 步骤 3: 根据 rect_count 选择最终的纸垛 ---
        selected_paper = None
        if not paper_candidates:
            print("未找到符合条件的纸垛候选者")
        elif rect_count == 1:
            selected_paper = max(paper_candidates, key=lambda p: p['width'])
        elif rect_count == 2:
            left_papers = [p for p in paper_candidates if p['center'][0] < midline_x]
            if left_papers:
                selected_paper = max(left_papers, key=lambda p: p['width'])
        elif rect_count == 3:
            right_papers = [p for p in paper_candidates if p['center'][0] > midline_x]
            if right_papers:
                selected_paper = max(right_papers, key=lambda p: p['width'])

        # --- 步骤 4: 处理和显示结果 (修改为只画三条线) ---
        x_offset, y_offset = 0, 0
        if selected_paper:
            geo = selected_paper['geo']
            left_line = geo['left_line']
            right_line = geo['right_line']
            bottom_line = geo['bottom_line']
            
            x_center, y_center = selected_paper['center']
            x_offset = x_center
            y_offset = h - y_center
            
            # 直接绘制检测到的三条原始线段
            # 蓝色侧边
            cv2.line(src_for_display, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 3)
            cv2.line(src_for_display, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255, 0, 0), 3)
            # 绿色底边
            cv2.line(src_for_display, (bottom_line[0], bottom_line[1]), (bottom_line[2], bottom_line[3]), (0, 255, 0), 3)
            
            # 绘制估算的中心点和坐标
            cv2.circle(src_for_display, (int(x_center), int(y_center)), 7, (0, 0, 255), -1)
            cv2.putText(src_for_display, f"XY: ({x_offset:.1f}, {y_offset:.1f})", 
                        (int(x_center) + 10, int(y_center)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        # 总是绘制中线
        cv2.line(src_for_display, (midline_x, 0), (midline_x, h), (0, 255, 255), 1)
        cv2.putText(src_for_display, f"Midline", 
                    (midline_x + 5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("Paper Detection", src_for_display)
        cv2.waitKey(1)

        return x_offset, y_offset