# from ultralytics import YOLO
import cv2
import time
# from CameraDetect import CameraDetect # 假设您的 CameraDetect 类在这个文件中
# from Uart import Uart

# --------------------------------------------------------------------------

class LogisticsRobot:
    def __init__(self, uart, camera_detector):
        self.uart = uart
        self.cam = camera_detector
        
        # self.up_layer = []
        # self.low_layer = []
        # self.zone_mapping = {}
        # self.box_mapping = {}
        # self.empty_positions = []
        # self.missing_num = 0

        ##test
        self.up_layer = [5, 4, 2]
        self.low_layer = [1, 6, 3]
        self.box_mapping = {'A': 1, 'B': 6, 'C': 3, 'D': 5, 'E': 4, 'F': 2}
        self.zone_mapping = {'a': 6, 'b': 2, 'c': 1, 'd': 5, 'e': None, 'f': 4}
        self.empty_positions = ['e']
        self.missing_num = 3

        # --- 定位相关配置 (保持不变) ---
        # 追踪哪些纸垛区域是存在的，且尚未被货箱覆盖
        self.available_localization_zones = set()
        # 将目标区域映射到其物理分组，用于寻找定位点
        self.localization_groups = {
            'a': ['a'], 
            'b': ['b', 'c'], 'c': ['b', 'c'],
            'd': ['d', 'e'], 'e': ['d', 'e'], 
            'f': ['f']
        }
        # 将具体的纸垛位置映射到 locate_paper() 需要的 rect_count 参数
        self.zone_to_rect_count = {
            'a': 1, 'f': 1,  # 单纸垛组
            'b': 0, 'd': 0,  # 双纸垛组的左侧纸垛
            'c': 2, 'e': 2   # 双纸垛组的右侧纸垛
        }

        # 定位成功的偏移量阈值
        self.LOCATE_X_THRESHOLD = 5.0
        self.LOCATE_Y_THRESHOLD = 5.0
        # 最大定位尝试次数
        self.MAX_LOCATE_ATTEMPTS = 5


        # --- 新增: 物理区域代号和状态 ---
        # 逻辑区域 -> 物理区域代号
        self.zone_to_physical_area_id = {
            'a': 1, 'b': 2, 'c': 2, 'd': 3, 'e': 3, 'f': 4
        }
        # 记录小车当前所在的物理区域代号，0代表起始区
        self.current_physical_area_id = 0 


    # --- 在扫描后初始化机器人状态的方法 ---
    def initialize_state_after_scan(self):
        """
        根据初始扫描结果，初始化可用于定位的区域集合。
        此方法应该在 self.zone_mapping 和 self.empty_positions 被填充后【调用一次】。
        """
        # (此函数保持不变)
        self.available_localization_zones = set(self.zone_mapping.keys()) - set(self.empty_positions)
        print(f"状态初始化完成。可定位区域: {sorted(list(self.available_localization_zones))}")

    # --- 重构: 移动函数 ---
    def move_to_physical_area(self, target_area_id):
        """
        根据新的通信协议控制小车移动到指定的物理区域。
        :param target_area_id: 目标物理区域代号 (1-4)
        """
        # if self.current_physical_area_id == target_area_id:
        #     print(f"信息: 小车已在目标区域 {target_area_id}，无需移动。")
        #     return True
            
        # 根据协议计算指令
        command = (self.current_physical_area_id << 4) | target_area_id
        print(f"\n-->> 指令: 从区域 {self.current_physical_area_id} 移动到区域 {target_area_id}。发送指令: {hex(command)}")
        self.uart.send_move_command(command)
        # # 3. 进入异步等待循环
        # print(f"信息: 等待硬件进入纸垛定位模式 (uart.current_task == Task.LOCATE_paper)...")
        # start_time = time.time()
        # timeout_seconds = 20  # 移动的超时时间可以设置长一些

        # # 循环等待，直到接收线程将 current_task 更新为 LOCATE_paper
        # while self.uart.current_task != self.uart.Task.LOCATE_paper:
        #     # 检查是否超时
        #     if time.time() - start_time > timeout_seconds:
        #         print(f"!!! 移动失败：超过 {timeout_seconds} 秒未收到进入“纸垛定位”模式的信号。")
        #         return False

        #     # # 在等待时，主线程可以短暂休眠，降低CPU占用
        #     # time.sleep(0.05)
        
        # 4. 循环结束，意味着硬件已经准备好进行纸垛定位
        print(f"信息: 移动完成！已收到进入“纸垛定位”模式的信号。当前位置更新为区域 {target_area_id}。")
        self.current_physical_area_id = target_area_id
        
        return True

    def place_box(self, box_num, zone, layer, box_key, put_down_side, is_stacking=False):
        # (此函数保持不变)
        placement_height = 'high' if is_stacking else 'low'
        placement_type = "堆叠放置" if is_stacking else "直接放置"
        
        if is_stacking:
            print(f"** 特殊操作: 将空位箱 {box_num}(车上{box_key}) 堆叠到 {zone} 区。 **")
        else:
            print(f"准备将 {box_num} 号货箱(车上{box_key}) 放置到 {zone} 区。")

        command = {'box_key': box_key, 'put_down_side': put_down_side, 'height': placement_height}
        print(f"  - 来源: {layer}层, 类型: {placement_type}, 指令: {command}")

        if self.uart.send_place_command(command):
            print(f"  - 结果: 成功放置 {box_num} 号货箱。\n")
            return True
        else:
            print(f"  - 结果: 放置 {box_num} 号货箱失败。\n")
            return False

    # #空纸垛定位
    # def _perform_localization(self, target_zone):
    #     """
    #     在放置到 target_zone 之前，执行视觉定位。
    #     采用异步等待模式，由 UART 接收线程设置事件来终止。
    #     """
    #     print(f"--- 开始为放置到 {target_zone} 进行定位 ---")
        
    #     # 1. 确定使用哪个物理分组进行定位
    #     group_zones = self.localization_groups.get(target_zone)
    #     # 2. 从该分组中找到一个仍然可用的定位点
    #     usable_zones_in_group = self.available_localization_zones.intersection(group_zones)
        
    #     if not usable_zones_in_group:
    #         print(f"!!! 严重错误: 定位组 {group_zones} 内已无任何可用定位点！")
    #         return False
    #     # 策略：选择组内任意一个可用的点进行定位（例如，第一个）
    #     localization_zone = list(usable_zones_in_group)[0]
    #     rect_count_param = self.zone_to_rect_count[localization_zone]
    #     print(f"信息: 目标 {target_zone} 属 {group_zones} 组，选择 {localization_zone} 作为参照物 (rect_count={rect_count_param})")

    #     # --- 重构: 执行异步定位闭环 ---

    #     # 2. 【关键】在开始定位前，重置 Uart 中的标志位
    #     #    这确保我们等待的是本次任务的完成信号，而不是上一次的。
    #     #    假设 flag_paper 为 1 代表完成，0 代表进行中。
    #     print("信息: 重置定位完成标志位 self.uart.flag_paper = 0")
    #     self.uart.flag_paper = 0  # 假设您可以在 Uart 类外部修改这个值

    #     print("信息: 进入定位模式，持续发送坐标，等待硬件完成信号 (uart.flag_paper == 1)...")
        
    #     # 3. 使用 while 循环，直到 uart 的标志位变为 1
    #     #    增加超时机制，防止因通信问题导致无限等待
    #     start_time = time.time()
    #     timeout_seconds = 15 # 设置一个合理的超时时间，例如15秒

    #     while self.uart.flag_paper != 1:
    #         # 检查是否超时
    #         if time.time() - start_time > timeout_seconds:
    #             print(f"!!! 定位失败：超过 {timeout_seconds} 秒，uart.flag_paper 仍不为 1。")
    #             # 在退出前，最好也重置一下标志位，避免影响下次
    #             self.uart.flag_paper = 0
    #             return False

    #         # 在循环中，持续调用相机检测并发送坐标
    #         x_offset, y_offset = self.cam.locate_paper(rect_count=rect_count_param)
            
    #         # 持续发送定位指令
    #         self.uart.send_locate_command(x_offset, y_offset)
            
    #         # 短暂延时，避免CPU占用过高和串口发送过于频繁
    #         time.sleep(0.001) 

    #     # 4. 循环结束，意味着 self.uart.flag_paper 已经被接收线程修改为 1
    #     print("  定位成功！已收到硬件（STM32）的完成信号 (uart.flag_paper == 1)。")
        
    #     # 在成功后，也重置标志位，为下一次定位任务做准备
    #     self.uart.flag_paper = 0
        
    #     return True

    def _perform_localization(self, target_zone):
        """
        在放置到 target_zone 之前，执行基于AprilTag的视觉定位。
        采用异步等待模式，由 UART 接收线程设置事件来终止。
        """
        print(f"--- 开始为放置到 {target_zone} 进行AprilTag精确定位 ---")
        
        # 新的定位逻辑统一使用AprilTag，不再需要根据 target_zone 选择不同的参考物。
        
        # --- 执行异步定位闭环 ---
        # 1. 重置 Uart 中的完成标志位
        # print("信息: 重置定位完成标志位 self.uart.flag_paper = 0")
        self.uart.flag_paper = 0

        print("AprilTag定位,等待完成信号 (uart.flag_paper == 1)")
        
        # 2. 使用 while 循环，直到 uart 的标志位变为 1，并加入超时
        start_time = time.time()
        timeout_seconds = 15 # 定位超时时间

        while self.uart.flag_paper != 1:
            # 检查是否超时
            # if time.time() - start_time > timeout_seconds:
            #     print(f"!!! 定位失败：超过 {timeout_seconds} 秒，硬件未反馈完成信号。")
            #     self.uart.flag_paper = 0 # 退出前重置标志位
            #     return False

            # 调用新的AprilTag定位函数
            x_offset, y_offset = self.cam.locate_apriltag_2d()
            
            # 如果没有检测到标签，则不发送指令，继续下一次循环尝试
            if x_offset == 0 or y_offset == 0:
                # print("  - 未检测到AprilTag，继续搜索...") # 此消息可能会刷屏，可选择性开启
                # time.sleep(0.002) # 短暂等待，避免CPU空转和无意义的快速循环
                continue

            # 持续发送定位指令 
            self.uart.send_locate_command(x_offset, y_offset)
            
            # 短暂延时，避免串口发送过于频繁
            time.sleep(0.001)

        # 3. 循环结束，意味着 self.uart.flag_paper 已经被接收线程修改为 1
        print("  定位成功！已收到硬件（STM32）的完成信号 (uart.flag_paper == 1)。")
        
        # 成功后重置标志位，为下次任务做准备
        self.uart.flag_paper = 0
        
        return True

    # --- 重构: 将移动逻辑整合进单个放置执行中 ---
    def _execute_single_placement(self, plan, box_key, layer_name, placed_on_zones):
        box_num = plan['box_num']
        is_stacking = plan['is_stacking']
        
        if is_stacking:
            if not placed_on_zones:
                print(f"错误：无法堆叠货箱 {box_num}，因为还没有放置任何基础货箱！")
                return False
            # 策略：堆叠到该层最后一次成功放置的普通箱子上
            # `placed_on_zones` 是一个有序列表，最后一个元素就是最新放置的
            target_zone = placed_on_zones[-1]
            plan['target_zone'] = target_zone
        else:
            target_zone = plan['target_zone']

        # 1. 移动到目标物理区域
        target_physical_area = self.zone_to_physical_area_id[target_zone]
        if not self.move_to_physical_area(target_physical_area):
            return False # 如果移动失败，则取消本次放置

        time.sleep(0.01)

        put_down_side = self.zone_to_rect_count[target_zone]
        # 3. 执行放置动作
        success = self.place_box(
            box_num=box_num, zone=target_zone, layer=layer_name,
            box_key=box_key, put_down_side=put_down_side, is_stacking=is_stacking
        )

        # 2. 在放置前调用精确定位
        if not self._perform_localization(target_zone):
            print(f"!!! 由于定位失败，取消放置货箱 {box_num} 到 {target_zone}。")
            return False
        
        # put_down_side = self.zone_to_rect_count[target_zone]
        # # 3. 执行放置动作
        # success = self.place_box(
        #     box_num=box_num, zone=target_zone, layer=layer_name,
        #     box_key=box_key, put_down_side=put_down_side, is_stacking=is_stacking
        # )

        time.sleep(0.01)
        return success

    # --- 重构: 核心处理流程，整合了新的堆叠和移动逻辑 ---
    def _process_layer(self, layer_name, layer_boxes, all_placements, box_to_key, sort_reverse):
        print(f"\n========== 开始处理 {layer_name} 货箱 ==========")
        if not any(p['box_num'] in layer_boxes for p in all_placements):
            print(f"{layer_name}没有需要放置的货箱。")
            return

        zone_order = ['a', 'b', 'c', 'd', 'e', 'f']
        
        # 筛选出属于本层的普通任务和堆叠任务
        layer_placements = [p for p in all_placements if p['box_num'] in layer_boxes]
        normal_tasks = [p for p in layer_placements if not p['is_stacking']]
        stacking_tasks = [p for p in layer_placements if p['is_stacking']]
        
        # 对普通任务按指定顺序排序
        normal_tasks.sort(key=lambda p: zone_order.index(p['target_zone']), reverse=sort_reverse)
        
        # 合并任务，普通任务在前，堆叠任务在后
        sorted_tasks_for_layer = normal_tasks + stacking_tasks
        
        print(f"处理顺序: {'从右到左' if sort_reverse else '从左到右'}")
        
        # `placed_on_zones_this_layer` 用于为本层的堆叠任务提供目标
        placed_on_zones_this_layer = [] 
        
        for plan in sorted_tasks_for_layer:
            box_num = plan['box_num']
            
            # 动态检查货箱是否还在车上
            # 使用 `any` 检查，因为 `layer_boxes` 是可变的
            if not any(b == box_num for b in layer_boxes):
                continue

            box_key = box_to_key.get(box_num)
            
            # 执行单个放置（包含移动和定位）
            success = self._execute_single_placement(plan, box_key, layer_name, placed_on_zones_this_layer)
            
            if success:
                # 从车上移除已放置的货箱
                # 必须用 self.up_layer 或 self.low_layer，因为 layer_boxes 只是一个副本
                if box_num in self.up_layer: self.up_layer.remove(box_num)
                if box_num in self.low_layer: self.low_layer.remove(box_num)
                
                # 如果是普通放置，更新状态
                if not plan['is_stacking']:
                    target_zone = plan['target_zone']
                    # 记录该区域已被占用，用于堆叠
                    placed_on_zones_this_layer.append(target_zone)
                    # 从可定位区域中移除
                    self.available_localization_zones.discard(target_zone)
                    print(f"状态更新: 区域 {target_zone} 已被覆盖。可用定位点: {sorted(list(self.available_localization_zones))}")
            else:
                print(f"!!! 放置货箱 {box_num} 失败，终止 {layer_name} 的后续操作。")
                break

    # --- 重构: 总控流程 ---
    def execute_placement(self):
        # print("--- 开始执行放置策略 (最终版) ---")
        box_to_key = {v: k for k, v in self.box_mapping.items()}
        all_placements = self._generate_placement_plans()

        # 阶段一：处理上层 (从左到右)
        self._process_layer(
            layer_name="上层", layer_boxes=self.up_layer, 
            all_placements=all_placements, box_to_key=box_to_key, sort_reverse=False
        )
        
        # 阶段二：处理下层 (从右到左)
        self._process_layer(
            layer_name="下层", layer_boxes=self.low_layer, 
            all_placements=all_placements, box_to_key=box_to_key, sort_reverse=True
        )

        # 最终检查
        # print("\n--- 所有放置任务执行完毕 ---")
        if self.up_layer or self.low_layer:
            print(f"警告：仍有未放置的货箱！ 上层: {self.up_layer}, 下层: {self.low_layer}")
        else:
            print("所有货箱已成功放置！")
            # 可选：任务完成后，让小车返回起始区
            print("任务完成，返回起始区。")
            self.uart.send_move_command(0x66)

    def _generate_placement_plans(self):
        # (此函数保持不变)
        plans = []
        for zone, target_num in self.zone_mapping.items():
            if target_num is not None:
                plans.append({'box_num': target_num, 'target_zone': zone, 'is_stacking': False})
        if self.missing_num != 0:
            plans.append({'box_num': self.missing_num, 'target_zone': None, 'is_stacking': True})
        return plans

# ================== 测试代码 ==================
if __name__ == '__main__':
    # 1. 初始化
    mock_uart = MockUart()
    mock_cam = MockCameraDetect()
    robot = LogisticsRobot(mock_uart, mock_cam)

    # 2. 设置初始状态
    robot.up_layer = [1, 3, 5]
    robot.low_layer = [2, 4, 6]
    robot.box_mapping = {'A': 2, 'B': 4, 'C': 6, 'D': 1, 'E': 3, 'F': 5}
    robot.zone_mapping = {'a': 1, 'b': 3, 'c': 4, 'd': None, 'e': 6, 'f': 2}
    robot.empty_positions = ['d']
    robot.missing_num = 5

    print("======== 初始状态 ========")
    print(f"车上上层: {robot.up_layer}")
    print(f"车上下层: {robot.low_layer}")
    print(f"地上纸垛: {robot.zone_mapping}")
    print(f"空纸垛: {robot.empty_positions[0]}, 对应数字: {robot.missing_num}")
    print(f"小车初始位置: 区域 {robot.current_physical_area_id} (起始区)")
    print("=========================\n")
    
    # 3. 初始化定位状态
    robot.initialize_state_after_scan()

    # 4. 执行放置策略
    robot.execute_placement()