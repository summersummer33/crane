# import time
# from adafruit_servokit import ServoKit

# # 初始化PCA9685，16个通道
# kit = ServoKit(channels=16)

# # --- 这是关键部分 ---
# # 将通道0上的舵机指定为标准舵机
# servo_channel = 0
# my_servo = kit.servo[servo_channel]

# # 告诉库，这个舵机的活动范围是360度
# # 默认是180度，所以这一步很重要
# my_servo.actuation_range = 180

# # ----------------------

# print(f"准备控制360度位置舵机，连接在通道 {servo_channel} 上")
# print("按 Ctrl+C 停止程序")

# try:
#     while True:
#         print("转到 0 度")
#         my_servo.angle = 0
#         time.sleep(2)

#         print("转到 90 度")
#         my_servo.angle = 90
#         time.sleep(2)

#         print("转到 180 度")
#         my_servo.angle = 180
#         time.sleep(2)

#         # print("转到 270 度")
#         # my_servo.angle = 270
#         # time.sleep(2)

#         # print("转到 360 度 (或回到起点)")
#         # my_servo.angle = 360
#         # time.sleep(3)


# except KeyboardInterrupt:
#     # 程序结束时，可以将舵机转回一个安全位置，比如中间位置
#     print("\n程序停止，舵机转回180度。")
#     my_servo.angle = 180
#     time.sleep(1)






# from __future__ import division
# import time

# # Import the PCA9685 module.
# import Adafruit_PCA9685
# # Initialise the PCA9685 using the default address (0x40).
# pwm = Adafruit_PCA9685.PCA9685(busnum=1)
# # Configure min and max servo pulse lengths
# servo_min = 150
# # Min pulse length out of 4096
# servo_max = 600
# # Max pulse length out of 4096
# # Helper function to make setting a servo pulse width simpler.
# def set_servo_pulse(channel, pulse):
#     pulse_length = 1000000
#     # 1,000, 000 us per second
#     pulse_length //= 60
#     #60 Hz
#     print ('{0}us per period'.format(pulse_length))
#     pulse_length //= 4096
#     # 12 bits of resolution
#     print ('{0}us per bit'.format(pulse_length))
#     pulse *= 1000
#     pulse //= pulse_length
#     pwm.set_pwm(channel, 0, pulse)
    
# def set_servo_angle (channe1, angle):
#     angle=4096*((angle*11)+500)/20000
#     pwm.set_pwm(channe1,0,int(angle))
#     # Set frequency to 50hz, good for servos.

# pwm.set_pwm_freq (50)
# print (" Moving servo on channel 0, press CtrI-C to quit...")
# set_servo_angle (4, 50)
# time.sleep (1)
# set_servo_angle (5, 100)
# time.sleep (1)
# # pwm. set pwm (4, 0, 300)
# # time. sleep (1)
# # pwm. set pwm (5, 0, 300)
# # time. sleep (1)


# import time
# import Adafruit_PCA9685

# # --- 配置参数 ---
# SERVO_CHANNEL = 0      # 控制通道0
# FREQ = 50              # PWM频率 50Hz

# # --- 这是你通过校准得到的值！---
# PULSE_MIN = 550        # 0度对应的脉宽 (µs)
# PULSE_MAX = 2450       # 360度对应的脉宽 (µs)
# # -----------------------------

# # 初始化PCA9685
# pwm = Adafruit_PCA9685.PCA9685(busnum=1)
# # 设置频率
# pwm.set_pwm_freq(FREQ)

# def set_angle(channel, angle):
#     """
#     将 0-360 度的角度转换为PCA9685的tick值并设置舵机。
#     """
#     if not 0 <= angle <= 360:
#         raise ValueError("角度必须在 0 到 360 之间")

#     # 1. 计算目标脉宽 (µs)
#     pulse_range = PULSE_MAX - PULSE_MIN
#     pulse = PULSE_MIN + (angle / 360) * pulse_range

#     # 2. 将脉宽 (µs) 转换为 tick 值 (0-4095)
#     # PWM周期 = 1/FREQ 秒 = (1/FREQ) * 1,000,000 µs
#     period_us = 1000000.0 / FREQ
#     tick = int(pulse / period_us * 4096)
    
#     # 3. 设置PWM
#     pwm.set_pwm(channel, 0, tick)
#     print(f"设置角度: {angle}°, 对应脉宽: {pulse:.2f}µs, 对应tick: {tick}")


# # --- 主程序 ---
# print(f"准备在通道 {SERVO_CHANNEL} 上控制舵机...")
# try:
#     while True:
#         print("\n转到 0 度")
#         set_angle(SERVO_CHANNEL, 0)
#         time.sleep(2)

#         print("转到 90 度")
#         set_angle(SERVO_CHANNEL, 90)
#         time.sleep(2)

#         print("转到 180 度")
#         set_angle(SERVO_CHANNEL, 180)
#         time.sleep(2)

#         print("转到 270 度")
#         set_angle(SERVO_CHANNEL, 270)
#         time.sleep(2)

#         print("转到 360 度")
#         set_angle(SERVO_CHANNEL, 360)
#         time.sleep(3)

# except (KeyboardInterrupt, SystemExit):
#     # 程序退出时，让舵机回到中间位置
#     print("\n程序停止，舵机转回180度。")
#     set_angle(SERVO_CHANNEL, 180)
#     time.sleep(1)
#     # 释放舵机（可选，停止发送PWM信号）
#     # pwm.set_pwm(SERVO_CHANNEL, 0, 0)


import time
import Adafruit_PCA9685

# 初始化
pwm = Adafruit_PCA9685.PCA9685(busnum=1)
pwm.set_pwm_freq(49)

# 这个函数将微秒(us)转换为tick值
def pulse_to_tick(pulse_us):
    # 周期是 1/50Hz = 20ms = 20000us
    # 每个tick代表的微秒数 = 20000us / 4096 ticks
    tick_per_us = 4096 / 20000
    return int(pulse_us * tick_per_us)

print("舵机脉宽校准程序。")
print("输入一个脉宽值 (通常在 400-2600 之间) 来找到0度和360度的精确值。")
print("输入 'q' 退出。")

SERVO_CHANNEL = 0 # 我们要在通道0上校准

while True:
    try:
        val_str = input("输入脉宽值 (µs), e.g., 1500: ")
        if val_str.lower() == 'q':
            break
        
        pulse = int(val_str)
        tick = pulse_to_tick(pulse)
        pwm.set_pwm(SERVO_CHANNEL, 0, tick)
        print(f"发送脉宽: {pulse} µs, 对应的tick值是: {tick}")

    except (ValueError, KeyboardInterrupt):
        print("\n退出校准。")
        break