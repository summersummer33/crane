import cv2
import os
import time

# 摄像头参数配置
DEVICE_NUM = 0                  # /dev/video0 设备号
RESOLUTION = (640, 480)         # 视频分辨率
FPS = 15                        # 帧率（树莓派USB摄像头建议15-30）
VIDEO_FORMAT = "MJPG"           # 视频编码格式（树莓派支持的格式：MJPG/XVID）
OUTPUT_FOLDER = "captured_frames"  # 输出照片目录
FRAME_INTERVAL = 8             # 帧间隔（每30帧保存1张）
START_COUNTER = 1602             # 起始编号

# 使用时间戳生成唯一的视频文件名
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
OUTPUT_VIDEO = f"raspi_record_{timestamp}.avi"  # 输出视频文件名

# 初始化v4l2摄像头
cap = cv2.VideoCapture(DEVICE_NUM, cv2.CAP_V4L2)

# 设置摄像头参数（必须与摄像头支持的分辨率/帧率匹配）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
# cap.set(cv2.CAP_PROP_FPS, FPS)

# 验证参数是否设置成功
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"摄像头实际参数：{actual_width}x{actual_height} @ {actual_fps}fps")

# 初始化视频编码器
fourcc = cv2.VideoWriter_fourcc(*VIDEO_FORMAT)
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, RESOLUTION)

# 创建输出目录
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 状态显示参数
start_time = time.time()
frame_count = 0
save_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频帧获取失败")
            break
        
        # 写入视频文件
        out.write(frame)
        
        # 每间隔帧保存一次
        if frame_count % FRAME_INTERVAL == 0:
            filename = f"all_paper_{START_COUNTER + save_count:06d}.jpg"
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            cv2.imwrite(output_path, frame)
            print(f"已保存: {output_path}")
            save_count += 1
        
        # 实时显示视频
        cv2.imshow('Live Video', frame)
        
        frame_count += 1
        
        # 控制台状态显示（每秒更新）
        if frame_count % FPS == 0:
            duration = time.time() - start_time
            print(f"录制中... 时长: {duration:.1f}s, 帧数: {frame_count}")
            
        # 按Q键退出（需要焦点在OpenCV窗口）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("用户中断录制")

finally:
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # 输出统计信息
    duration = time.time() - start_time
    print(f"\n录制完成！视频文件保存为 {OUTPUT_VIDEO}")
    print(f"照片保存在 {OUTPUT_FOLDER} 目录下")
    print(f"总时长: {duration:.2f}s")
    print(f"总帧数: {frame_count}")
    print(f"实际帧率: {frame_count/duration:.2f}fps")
    print(f"共保存 {save_count} 张图片")