import sys
import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


#  GPIO12--PWM--右侧一列倒数第五个

# servo_pin 舵机信号线接树莓派GPIO17
def complex_change(servo_pin):
	GPIO.setup(servo_pin, GPIO.OUT, initial = False)
	p = GPIO.PWM(servo_pin, 50)  # 初始频率为50HZ
	p.start(angleToDutyCycle(105))  # 舵机初始化角度为90，p.start(5.833)
	sleep(1.5)

	p.ChangeDutyCycle(0)  # 清空当前占空比，使舵机停止抖动

	return p


# 水平旋转角度转换到PWM占空比
def angleToDutyCycle(angle):
	# 注意，这里舵机的范围 是0到180度，如果舵机的范围是0到270，则需要将180更换为270
	return 2.5 + (angle / 360.0 )* 10

def cleanup():
    """清理资源函数"""
    p.stop()  # 停止PWM输出
    GPIO.cleanup()  # 清理GPIO设置
    print("\nGPIO已清理，程序退出")

if __name__ == '__main__':
	try:
		p = complex_change(12)
		while True:
			# angle = int(input('水平旋转度数：'))
			# 17 水平转动
			# p = complex_change(12)
			# for i in range(0,5):
			# 	angle = i*90
			# 	p.ChangeDutyCycle(angleToDutyCycle(angle))
			# 	print("angle:",angle)
			# 	sleep(0.1)
			# 	p.ChangeDutyCycle(0)  # 清空当前占空比，使舵机停止抖动
			# 	sleep(1.5)
			# for i in range(4, -1, -1):
			# 	angle = i * 90
			# 	p.ChangeDutyCycle(angleToDutyCycle(angle))
			# 	print("angle:",angle)
			# 	sleep(0.1)
			# 	p.ChangeDutyCycle(0)  # 清空当前占空比，使舵机停止抖动
			# 	sleep(1.5)
				

			# sleep(0.1)
			p.ChangeDutyCycle(0)  # 清空当前占空比，使舵机停止抖动
	except KeyboardInterrupt:
        # 捕获Ctrl+C中断
		print("\n检测到Ctrl+C，正在清理资源...")
		cleanup()
		sys.exit(0)
	
	except Exception as e:
        # 捕获其他异常
		print(f"发生错误: {e}")
		cleanup()
		sys.exit(1)
		
    
# import RPi.GPIO as GPIO

# GPIO.setmode(GPIO.BCM)

# # 设置你使用过的所有引脚
# used_pins = [12]  # 添加所有你使用过的引脚号
# for pin in used_pins:
#     GPIO.setup(pin, GPIO.OUT)  # 或 GPIO.IN，根据实际情况

# # 现在清理这些引脚
# GPIO.cleanup()
# print("GPIO 清理完成")