import cv2
import time
from CameraDetect import CameraDetect
from Uart import Uart, Task
from LogisticsRobot_pro import LogisticsRobot
import RPi.GPIO as GPIO


#######舵机旋转
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def complex_change(servo_pin):
	GPIO.setup(servo_pin, GPIO.OUT, initial = False)
	p = GPIO.PWM(servo_pin, 50)  # 初始频率为50HZ
	p.start(angleToDutyCycle(90))  # 舵机初始化角度为90，p.start(5.833)
	time.sleep(1.5)

	p.ChangeDutyCycle(0)  # 清空当前占空比，使舵机停止抖动

	return p

def angleToDutyCycle(angle):
	# 注意，这里舵机的范围 是0到180度，如果舵机的范围是0到270，则需要将180更换为270
	return 2.5 + (angle / 360.0 )* 10


def main():
    uart = Uart()
    # p = complex_change(12)
    cam = CameraDetect(uart)
    cam._init_camera(1)  # 初始化摄像头
    fcn = LogisticsRobot(uart)
    # fcn.cam._init_camera(1) 
    angle = 0

    while True:
        if uart.current_task == Task.DETECT_box:
            fcn.up_layer, fcn.low_layer = cam.detect_boxes()

            p.ChangeDutyCycle(angleToDutyCycle(angle))
            time.sleep(0.1)
            p.ChangeDutyCycle(0)
        
        elif uart.current_task == Task.LOCATE_box:
            while uart.flag_paper == 0:
                x, y = cam.locate_box()
                uart.send_locate_command(x, y)

            p.ChangeDutyCycle(angleToDutyCycle(angle))
            time.sleep(0.1)
            p.ChangeDutyCycle(0)  
        
        elif uart.current_task == Task.DETECT_paper:
            cam.detect_zones(0)  #看左面

            p.ChangeDutyCycle(angleToDutyCycle(angle))
            time.sleep(0.1)
            p.ChangeDutyCycle(0)

            cam.detect_zones(1)  #看右面

            direction = fcn.match_zone()
            fcn.move_to_zone(direction)

            p.ChangeDutyCycle(angleToDutyCycle(angle))
            time.sleep(0.1)
            p.ChangeDutyCycle(0)
        
        elif uart.current_task == Task.LOCATE_paper:
            while uart.flag_paper > 1:
                x, y = cam.locate_paper(uart.flag_paper-1)
                uart.send_locate_command(x, y)
            # while True:
            #      x,y = cam.locate_paper(2)
            #      uart.send_locate_command(x,y)
            print("arrive::::::::::::::")

        elif uart.current_task == Task.PLACE_box:

            # fcn.execute_placement()
            fcn.initialize_state_after_scan()

            # 4. 执行放置策略
            fcn.execute_placement()

            uart.current_task = None








if __name__ == "__main__":
    main()