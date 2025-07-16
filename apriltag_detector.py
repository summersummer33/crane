import cv2
import numpy as np

# --- 1. 加载相机标定数据 ---
# 确保 'camera_calibration_data.npz' 文件在同一个目录下
try:
    calibration_data = np.load('camera_calibration_data.npz')
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']
except FileNotFoundError:
    print("错误：找不到相机标定文件 'camera_calibration_data.npz'。")
    print("请先运行相机标定程序。")
    exit()


# --- 2. 定义AprilTag字典和检测器参数 ---
# 选择你使用的AprilTag家族，例如DICT_APRILTAG_36h11
# 注意：在旧版OpenCV中，aruco字典可能在cv2.aruco模块下
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

# 创建检测器参数
aruco_params = cv2.aruco.DetectorParameters_create() 
# 【兼容性修改】使用 DetectorParameters_create() 函数，这是旧版本的API
# 你可以在这里设置参数，例如：
# aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX


# --- 3. 设置AprilTag的物理尺寸（单位：米） ---
MARKER_SIZE_M = 0.1 # 示例：边长为5厘米，请修改为你自己的精确值


# --- 4. 打开摄像头并开始检测 ---
# 尝试使用V4L2后端，这在Linux上更稳定，可能避免Segmentation Fault
cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
if not cap.isOpened():
    # 如果V4L2失败，回退到默认设置
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

print("摄像头已启动，开始检测AprilTag...")
print("按 'q' 键退出。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法捕获帧，退出...")
        break
    
    # 将图像转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 【核心修改】直接调用检测函数，而不是创建和使用Detector对象
    # 这是兼容旧版OpenCV的写法
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray, 
        aruco_dict, 
        parameters=aruco_params
    )
    
    # 如果检测到标记
    if ids is not None and len(ids) > 0:
        # 绘制检测到的标记
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # 估计每个标记的位姿
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_M, mtx, dist)
        
        # 在图像上为每个标记绘制坐标轴
        for i in range(len(ids)):
            # 旧版OpenCV的drawFrameAxes可能不存在，我们使用drawAxis
            # 如果drawFrameAxes报错，请使用下面的drawAxis
            try:
                cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], 0.03) # 坐标轴长度为3厘米
            except AttributeError:
                # 兼容更老版本的OpenCV
                cv2.aruco.drawAxis(frame, mtx, dist, rvecs[i], tvecs[i], 0.03)

            
            # 打印位姿信息
            distance = np.linalg.norm(tvecs[i][0]) # 更简洁的计算距离方式
            print("angle:???",rvecs[i])
            print(f"ID: {ids[i][0]} | Distance: {distance:.2f} m | Position (x,y,z): {np.round(tvecs[i][0], 2)}")

    # 显示结果
    cv2.imshow('AprilTag Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()