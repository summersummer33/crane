import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


#  GPIO12--PWM--右侧一列倒数第五个

# servo_pin 舵机信号线接树莓派GPIO17
def complex_change(servo_pin):
	GPIO.setup(servo_pin, GPIO.OUT, initial = False)
	p = GPIO.PWM(servo_pin, 50)  # 初始频率为50HZ
	p.start(angleToDutyCycle(90))  # 舵机初始化角度为90，p.start(5.833)
	sleep(1.5)

	p.ChangeDutyCycle(0)  # 清空当前占空比，使舵机停止抖动

	return p


# 水平旋转角度转换到PWM占空比
def angleToDutyCycle(angle):
	# 注意，这里舵机的范围 是0到180度，如果舵机的范围是0到270，则需要将180更换为270
	return 2.5 + (angle / 360.0 )* 10


# # 垂直旋转角度转换到PWM占空比
# def angleToDutyCycleVertical(angle):
# 	return 2.5+(angle / 180.0 )* 10

# 270度垂直旋转角度转换到PWM占空比
# 190度 回正
# 90度 下落
#def angleToDutyCycleVertical(angle):
#	return (angle / 180.0 )* 10

if __name__ == '__main__':
	p = complex_change(12)
	while True:
		# angle = int(input('水平旋转度数：'))
		# 17 水平转动
		# p = complex_change(12)
		# sleep(1)
		# p.ChangeDutyCycle(angleToDutyCycle(0))
		# sleep(1)
		# p.ChangeDutyCycle(angleToDutyCycle(90))
		# sleep(1)
		# p.ChangeDutyCycle(angleToDutyCycle(180))
		# sleep(1)
		# p.ChangeDutyCycle(angleToDutyCycle(270))
		# sleep(1)
		# p.ChangeDutyCycle(angleToDutyCycle(360))
		# sleep(1)
		# p.ChangeDutyCycle(angleToDutyCycle(270))
		# sleep(1)
		# p.ChangeDutyCycle(angleToDutyCycle(180))
		# sleep(1)
		for i in range(0,5):
			angle = i*90
			p.ChangeDutyCycle(angleToDutyCycle(angle))
			print("angle:",angle)
			sleep(0.1)
			p.ChangeDutyCycle(0)  # 清空当前占空比，使舵机停止抖动
			sleep(1.5)
		for i in range(4, -1, -1):
			angle = i * 90
			p.ChangeDutyCycle(angleToDutyCycle(angle))
			print("angle:",angle)
			sleep(0.1)
			p.ChangeDutyCycle(0)  # 清空当前占空比，使舵机停止抖动
			sleep(1.5)
				



		# # 18 垂直方向回正
		# p = complex_change(18)
		# # 90度 下落
		# p.ChangeDutyCycle(angleToDutyCycleVertical(90))
		# sleep(0.5)
		# # 190度 回正
		# p.ChangeDutyCycle(angleToDutyCycleVertical(190))


		# sleep(0.1)
		p.ChangeDutyCycle(0)  # 清空当前占空比，使舵机停止抖动
    
