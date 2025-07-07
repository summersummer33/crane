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

        # 初始化YOLO检测器（只初始化一次）
        if cam_mode == 0:
            self._init_detector()


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
                self.cap_detect = cv2.VideoCapture(0,cv2.CAP_V4L2) #V4l2树莓派
                if not self.cap_detect.isOpened():
                    raise ValueError("camera not open")
                self.cap_detect.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) 
                # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
                for _ in range(5):
                    ret, _ = self.cap_detect.read()
                    time.sleep(0.01)

            except Exception as e:
                print(f"camera init error: {e}")
                raise
            
        elif(cam_mode ==1):  #定位摄像头

            self.cap_locate = cv2.VideoCapture(0,cv2.CAP_V4L2) 
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

        
        # 图像预处理
        # gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

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
        cv2.imshow("bright", bright)
        cv2.imshow("mask", mask3)
        cv2.imshow("opened", opened)
        # cv2.imshow("edges", edges)
        
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
            if area > 5000:  # 最小面积阈值
                
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

            
            large_contours_ = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 400 :
                    large_contours_.append(contour)
                    x_b1, y_b1, w_b1, h_b1 = cv2.boundingRect(contour)
                    cv2.rectangle(src, (x_b1, y_b1), (x_b1 + w_b1, y_b1 + h_b1), (0, 0, 255), 2)
            if large_contours_:
                merged_contour_b = np.vstack(large_contours_)
                x_b, y_b, w_b, h_b = cv2.boundingRect(merged_contour_b)
                cv2.rectangle(src, (x_b, y_b), (x_b + w_b, y_b + h_b), (255, 0, 0), 2)
                cv2.imshow("src:", src)
        
        # if debug and locations:
        cv2.imshow("Rectangle Detection", self.frame)
        cv2.waitKey(1)

        return x_offset, y_offset

    
    def locate_paper(self):
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