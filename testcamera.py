import cv2
import os
import time

# 摄像头参数配置
DEVICE_NUM = 0                  # /dev/video0 设备号
RESOLUTION = (640, 480)         # 视频分辨率
FPS = 30                        # 帧率（树莓派USB摄像头建议15-30）
VIDEO_FORMAT = "MJPG"           # 视频编码格式（树莓派支持的格式：MJPG/XVID）
OUTPUT_VIDEO = "raspi_record.avi"  # 输出视频文件名
OUTPUT_FOLDER = "captured_frames"  # 输出照片目录
FRAME_INTERVAL = 15             # 帧间隔（每30帧保存1张）
START_COUNTER = 297             # 起始编号

# 初始化v4l2摄像头
cap = cv2.VideoCapture(DEVICE_NUM, cv2.CAP_V4L2)

# 设置摄像头参数（必须与摄像头支持的分辨率/帧率匹配）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
cap.set(cv2.CAP_PROP_FPS, FPS)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频帧获取失败")
            break
        
        # 实时显示视频
        cv2.imshow('Live Video', frame)

            
        # 按Q键退出（需要焦点在OpenCV窗口）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("用户中断录制")

finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
