import cv2
import time
from CameraDetect import CameraDetect
from Uart import Uart, Task
from LogisticsRobot_pro import LogisticsRobot
import RPi.GPIO as GPIO


#######舵机旋转
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

angle_detect_box = 117
angle_detect_paper_left = 170
angle_detect_paper_right = 70
angle_locate = 297



def complex_change(servo_pin):
	GPIO.setup(servo_pin, GPIO.OUT, initial = False)
	p = GPIO.PWM(servo_pin, 50)  # 初始频率为50HZ
	p.start(angleToDutyCycle(angle_detect_box))  # 舵机初始化角度为90，p.start(5.833)
	time.sleep(1)

	p.ChangeDutyCycle(0)  # 清空当前占空比，使舵机停止抖动

	return p

def angleToDutyCycle(angle):
	# 注意，这里舵机的范围 是0到180度，如果舵机的范围是0到270，则需要将180更换为270
	return 2.5 + (angle / 360.0 )* 10


def main():
    uart = Uart()
    p = complex_change(18)
    cam = CameraDetect(uart)
    cam._init_camera(1)  # 初始化摄像头
    fcn = LogisticsRobot(uart,cam)
    print("initial")
    

    while True:
        if uart.current_task == Task.DETECT_box:
            # fcn.up_layer, fcn.low_layer, fcn.box_mapping = cam.detect_boxes()
            print("detect_box")
            p.ChangeDutyCycle(angleToDutyCycle(angle_locate))
            time.sleep(0.5)
            p.ChangeDutyCycle(0)
            uart.current_task = None
        
        elif uart.current_task == Task.LOCATE_box:
            print("locate_box")
            while uart.flag_box == 0:
                x, y = cam.locate_apriltag_2d()
                uart.send_locate_command(x, y)
                time.sleep(0.005)
                

            p.ChangeDutyCycle(angleToDutyCycle(angle_detect_paper_left))
            time.sleep(0.2)
            p.ChangeDutyCycle(0)  
            uart.current_task = None
            uart.flag_box = 0
        
        elif uart.current_task == Task.DETECT_paper:
            print("detect_paper")
            # cam.detect_zones(0)  #看左面

            p.ChangeDutyCycle(angleToDutyCycle(angle_detect_paper_right))
            time.sleep(0.5)
            p.ChangeDutyCycle(0)

            # fcn.zone_mapping, fcn.missing_num, fcn.empty_positions = cam.detect_zones(1)  #看右面
            time.sleep(1)

            p.ChangeDutyCycle(angleToDutyCycle(angle_locate))
            time.sleep(0.5)
            p.ChangeDutyCycle(0)
            uart.current_task = None

            fcn.initialize_state_after_scan()

            # 4. 执行放置策略
            fcn.execute_placement()

            uart.current_task = None
            uart.flag_paper = 0
        
        # elif uart.current_task == Task.LOCATE_paper:
        #     while uart.flag_paper == 0:
        #         x, y = cam.locate_apriltag_2d()
        #         uart.send_locate_command(x, y)
        #     # while uart.flag_paper > 1:
        #     #     x, y = cam.locate_paper(uart.flag_paper-1)
        #     #     uart.send_locate_command(x, y)
        #     # while True:
        #     #      x,y = cam.locate_paper(2)
        #     #      uart.send_locate_command(x,y)
        #     print("arrive::::::::::::::")
        #     uart.current_task = None
        #     uart.flag_paper = 0

        # elif uart.current_task == Task.PLACE_box:

        #     fcn.initialize_state_after_scan()

        #     # 4. 执行放置策略
        #     fcn.execute_placement()

        #     uart.current_task = None
        #     uart.flag_paper = 0








if __name__ == "__main__":
    main()