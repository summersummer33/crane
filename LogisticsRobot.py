# from ultralytics import YOLO
import cv2
import time
from CameraDetect import CameraDetect
from Uart import Uart

class LogisticsRobot:
    def __init__(self, uart):
        self.uart = uart
        self.up_layer = []    # 车上（上层）
        self.low_layer = []   # 车上（下层，也是货架的下层）
        # 存储置物区映射关系
        # self.cam = CameraDetect()  
        self.zone_mapping = {}  #位于场地中间面对纸垛从左到右abcdef
        self.box_mapping = {}  #位于场地中间面对货架先上后下从左到右ABCDEF
        self.empty_positions = []
        self.missing_num = None
        
        # 坐标校准参数（需要根据实际场地测量）
        self.zone_coordinates = {
            'left': {
                'a': (0.2, 0.3),
                'b': (0.2, 0.5),
                'c': (0.2, 0.7)
            },
            'right': {
                'd': (0.8, 0.3),
                'e': (0.8, 0.5),
                'f': (0.8, 0.7)
            }
        }

    def move_to_zone(self, direction):
        """控制小车移动到置物区"""
        if direction == "left":
            print("正在向左移动...")
            self.uart.send_move_command(0)  # 发送左移指令
        else:
            print("正在向右移动...")
            self.uart.send_move_command(1)  # 发送右移指令
        
        # # 等待移动完成信号（需根据实际协议实现）
        # if self.uart.wait_for_signal(taskid=10, state=1, timeout=10):
        #     print("移动完成")
        #     return True
        # else:
        #     print("移动超时")
        #     return False

    def match_zone(self):
        # 获取左右两侧纸垛的完整信息（包含空纸垛的已知数字）
        left_nums = [self.zone_mapping[zone] for zone in ['a', 'b', 'c']]
        right_nums = [self.zone_mapping[zone] for zone in ['d', 'e', 'f']]
                
        # 计算匹配度（不包含空纸垛数字的匹配）
        left_score = sum(1 for num in self.up_layer if num in left_nums)
        right_score = sum(1 for num in self.up_layer if num in right_nums)
        
        print("\n匹配逻辑：")
        print(f"左侧纸垛值：{left_nums} | 空纸垛数字：{self.missing_num}")
        print(f"右侧纸垛值：{right_nums}")
        print(f"上层目标数字：{self.up_layer}")
        print(f"左侧匹配度：{left_score}")
        print(f"右侧匹配度：{right_score}")
        
        if left_score >= right_score:
            print("--> 选择左侧放置")
            return "left"
        else:
            print("--> 选择右侧放置")
            return "right"
        
    def execute_placement(self):
        """总控放置流程"""
        initial_side = self.match_zone()
        other_side = "right" if initial_side == "left" else "left"

        # 处理初始方向侧
        self.process_side(initial_side)
        
        # 处理另一侧
        self.process_side(other_side)
        
        # 处理剩余货箱
        if self.low_layer or self.up_layer:
            print("仍有剩余货箱，返回初始方向处理")
            self.process_side(initial_side)

        # 最终检查
        if self.low_layer or self.up_layer:
            print("警告：仍有未放置的货箱！", self.low_layer, self.up_layer)
        else:
            print("所有货箱已成功放置")

    def process_side(self, direction):
        """处理指定侧的货箱放置"""
        print(f"\n=== 开始处理 {direction}侧 ===")
        if not self.move_to_zone(direction):
            return

        # 先处理上层货箱
        print("处理上层货箱...")
        self.process_layer(direction, "up")
        
        if not self.up_layer:
            # 再处理下层货箱
            print("处理下层货箱...")
            self.process_layer(direction, "low")

    def process_layer(self, direction, layer):
        """处理指定层的货箱放置"""
        current_layer = self.low_layer if layer == "low" else self.up_layer
        if not current_layer:
            return

        print(f"当前处理层：{layer}层，剩余货箱：{current_layer}")
        
        # 获取该侧所有zone信息
        side_zones = list(self.zone_coordinates[direction].keys())
        zone_info = {zone: self.zone_mapping.get(zone) for zone in side_zones}

        # 处理直接匹配的货箱
        placed_zones = set()
        to_remove = []
        for zone in side_zones:
            zone_num = zone_info[zone]
            if zone_num and zone_num in current_layer:
                if self.place_box(zone_num, zone, direction, layer):
                    to_remove.append(zone_num)
                    placed_zones.add(zone)
        
        # 移除已处理的货箱
        for num in to_remove:
            while num in current_layer:
                current_layer.remove(num)

        # 处理缺失数字的货箱
        missing_box = next((b for b in current_layer if b == self.missing_num), None)
        if missing_box is not None:
            # 寻找可放置的非空已放置区域
            available_zones = [z for z in side_zones if zone_info[z] is not None and z in placed_zones]
            if available_zones:
                target_zone = available_zones[0]
                if self.place_box(missing_box, target_zone, direction, layer):
                    current_layer.remove(missing_box)

    def place_box(self, box_num, zone, direction, layer):
        """执行实际的放置操作"""
        # 获取物理坐标
        x_percent, y_percent = self.zone_coordinates[direction][zone]
        physical_x, physical_y = self.map_physical_coordinates(x_percent, y_percent)
        
        # 构造放置命令（根据实际通信协议调整）
        # command = {
        #     'x': physical_x,
        #     'y': physical_y,
        #     'height': 'high' if layer == "second" else 'low',
        #     'box_num': box_num
        # }
        command = {
            'zone': 0 if direction == "left" else 1,  # 0 表示左侧，1 表示右侧
            'position': zone,  # zone 是 'a', 'b', 'c', 'd', 'e', 'f' 中的一个
            'height': 'high' if layer == "second" else 'low',
            'box_num': box_num
        }
        
        print(f"正在放置 {box_num} 号货箱到 {direction}侧{zone}区，"
              f"物理坐标({physical_x:.0f}, {physical_y:.0f})，"
              f"层级：{layer}")

        # 发送放置指令（示例代码）
        if self.uart.send_place_command(command):
            print(f"成功放置 {box_num} 号货箱")

            # 更新box_mapping，将对应的数字设置为None
            for pos, num in self.box_mapping.items():
                if num == box_num:
                    self.box_mapping[pos] = None
                    print(f"更新box_mapping：位置 {pos} 的货箱 {box_num} 已放置并标记为None")
                    break  # 假设编号唯一

            return True
        print(f"放置 {box_num} 号货箱失败")
        return False

    # def locate_paper(self):
    #     x, y = self.cam.locate_paper()
    #     self.uart.send_locate_command(x, y)

    # def locate_box(self):
    #     x, y = self.cam.locate_box()
    #     self.uart.send_locate_command(x, y)
   

