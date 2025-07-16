import time
import serial
import struct
import threading
from enum import IntEnum

class RobotCommand(IntEnum):
    MOVE = 0x01
    GRAB = 0x02
    PLACE = 0x03
    LOCATE = 0x04

class Task(IntEnum):
    DETECT_box = 0x05
    DETECT_paper = 0x06
    LOCATE_box = 0x07
    LOCATE_paper = 0x08
    PLACE_box = 0x09


class Uart:
    def __init__(self, port='/dev/ttyAMA2', baudrate=115200):
        self.serial = serial.Serial(port, baudrate, timeout=0.01)
        self.lock = threading.Lock()
        self.running = True
        self.current_task = None
        self.flag_paper = 0 #纸垛到位标志位
        self.flag_box = 0 #箱子到位标志位
        
        # 启动接收线程
        self.receive_thread = threading.Thread(target=self._receive_handler)
        self.receive_thread.daemon = True
        self.receive_thread.start()

    # def _pack_command(self, cmd_type, data):
    #     """协议格式: [STX][CMD][LEN][DATA...][ETX]"""
    #     stx = b'\x02'
    #     etx = b'\x03'
    #     cmd_byte = struct.pack('>B', cmd_type)
    #     int_data = [int(x) for x in data]  # 强制转换为整数
    #     data_bytes = struct.pack('>'+'H'*len(int_data), *int_data)
    #     # data_bytes = struct.pack('!'+'f'*len(data), *data)
    #     # length = struct.pack('B', len(data_bytes))
    #     pack_data = stx + cmd_byte + data_bytes + etx
    #     # print("pack_data:",pack_data)
    #     return pack_data

    def _pack_command(self, cmd_type, data):
        """协议格式: [STX][CMD][LEN][DATA...][ETX]
        支持处理十六进制字符串形式的data参数
        """
        stx = b'\x02'
        etx = b'\x03'
        cmd_byte = struct.pack('>B', cmd_type)
        # 处理可能包含十六进制字符串的数据
        int_data = []
        for x in data:
            if isinstance(x, str):
                # 如果是字符串，尝试解析为十六进制(以0x开头)或十进制
                if x.lower().startswith('0x'):
                    int_data.append(int(x, 16))
                else:
                    int_data.append(int(x))
            else:
                # 如果不是字符串，直接转换为整数
                int_data.append(int(x))

        # 动态选择数据打包格式
        if cmd_type == RobotCommand.LOCATE:  # LOCATE 指令，使用 2 字节 (H)
            data_format = '>' + 'H' * len(int_data)  # 大端序，无符号短整型
        else:                 # 其他指令，使用 1 字节 (B)
            data_format = '>' + 'B' * len(int_data)  # 大端序，无符号字节

        data_bytes = struct.pack(data_format, *int_data)  # 打包数据
        pack_data = stx + cmd_byte + data_bytes + etx
        return pack_data
    
    def send_move_command(self, direction):
        """
        发送移动指令
        """
        with self.lock:
            cmd_data = [direction, 0, 0, 0]
            packet = self._pack_command(RobotCommand.MOVE, cmd_data)
            self.serial.write(packet)
            print("packet:",packet)
            print(f"发送移动指令")

    # def send_grab_command(self, x, y, layer):
    #     """
    #     发送抓取指令
    #     :param x: 货架X坐标 (mm)
    #     :param y: 货架Y坐标 (mm)
    #     :param layer: 0-下层, 1-上层
    #     """
    #     with self.lock:
    #         cmd_data = [x, y, layer]
    #         packet = self._pack_command(RobotCommand.GRAB, cmd_data)
    #         self.serial.write(packet)
    #         print(f"发送抓取指令: X{x} Y{y} 层{layer}")

    def send_place_command(self, command):
        """
        发送包含高度信息的放置指令。
        :param command: 一个包含放置信息的字典，格式如下：
        """
        # --- 数据转换与校验 ---
        boxkey_char = command.get('box_key')
        put_down_side = command.get('put_down_side')
        height_char = command.get('height')

        # 处理 box_key 映射
        if boxkey_char in ['A', 'B', 'C', 'D', 'E', 'F']:
            box_id = ord(boxkey_char) - ord('A') + 1  # 'A'->1, 'B'->2, ..., 'F'->6

        if height_char == 'high':
            height_id = 1 # 1 代表高位
        elif height_char == 'low':
            height_id = 0 # 0 代表低位
        else:
            print(f"错误：无效的高度信息 '{height_char}'")
            return False

        with self.lock:
            cmd_data = [box_id, put_down_side, height_id, 0]
            packet = self._pack_command(RobotCommand.PLACE, cmd_data)
            print("place_packet:",packet)
            self.serial.write(packet) 
            
            # # 5. 打印详细的日志信息
            # height_text = "高位（叠放）" if height_id == 0 else "低位（直接放置）"
            # print(f"位置={zone_char}" f"高度={height_text}")
        
        return True # 假设发送成功

    def send_locate_command(self, x, y):
        """
        发送定位坐标指令
        :param x: 物块的X坐标 (mm)
        :param y: 物块的Y坐标 (mm)
        """
        with self.lock:
            cmd_data = [x, y]
            packet = self._pack_command(RobotCommand.LOCATE, cmd_data)
            # print(f"packet: { [hex(b) for b in packet] }") 
            self.serial.write(packet)
            # if x != 0 and y != 0 :
            #     print(f"发送定位坐标指令: X{x} Y{y}")

    def _receive_handler(self):
        """接收处理线程"""
        while self.running:
            if self.serial.in_waiting >= 4:  # 最小有效数据长度
                header = self.serial.read(1)
                if header == b'\x02':
                    # cmd = ord(self.serial.read(1))
                    # length = ord(self.serial.read(1))
                    # data = self.serial.read(length)
                    data = self.serial.read(2)
                    etx = self.serial.read(1)
                    print("recv : ",header+data+etx)
                    
                    if etx == b'\x03':
                        # self._process_received(cmd, data)
                        self.current_task = data[0]
                        if self.current_task == Task.LOCATE_box:
                            self.flag_box = data[1]
                        elif self.current_task == Task.LOCATE_paper:
                            self.flag_paper = data[1]
            time.sleep(0.04)  # 添加短暂延时，避免过于频繁的读取   


    # def _process_received(self, cmd, data):
    #     """处理接收到的数据 - 简化版，仅处理错误响应"""
    #     try:
    #         # 只处理错误响应
    #         if cmd == 0xFF:  # 错误命令码
    #             error_code = struct.unpack('B', data)[0]
    #             print(f"接收到错误代码: 0x{error_code:02X}")
    #     except Exception as e:
    #         print(f"数据处理错误: {str(e)}")

    def emergency_stop(self):
        """紧急停止"""
        with self.lock:
            self.serial.write(b'\x02\xFF\x00\x03')
            print("发送紧急停止指令")

    # def close(self):
    #     self.running = False
    #     self.receive_thread.join()
    #     self.serial.close()

    def close(self):
        self.running = False
        if self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1.0)  # 安全等待线程结束
        self.serial.close()
        print("串口已关闭")
