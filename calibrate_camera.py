import cv2
import numpy as np
import glob

# --- 1. 定义棋盘格参数 ---
# 棋盘格内部角点的数量，例如：9x6的棋盘格，corners_cols=9, corners_rows=6
CHECKERBOARD = (9, 6) 
# 每个棋盘格方块的物理尺寸（例如，毫米）。自己用尺子量一下！
SQUARE_SIZE_MM = 19.8 

# --- 2. 设置标定标准和图像路径 ---
# 终止标准，用于亚像素角点精细化
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备3D世界坐标点，如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM # 转换到真实世界的毫米单位

# 用于存储所有图像的世界坐标点和图像坐标点
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# --- 3. 实时捕捉图像并寻找角点 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
    
print("按 's' 键保存当前帧用于标定，按 'q' 键退出并计算。")
print("请从不同角度和距离拍摄棋盘格，建议15-20张。")

img_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 寻找棋盘格角点
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    # 如果找到了，绘制并显示
    if ret_corners:
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret_corners)
    
    cv2.imshow('Calibration Capture', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and ret_corners:
        # 亚像素级精确化
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        img_count += 1
        print(f"成功保存第 {img_count} 张图像数据。")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- 4. 执行相机标定 ---
if img_count < 10:
    print("标定所需图像不足，请重新运行并拍摄更多图像。")
else:
    print("开始计算相机参数...")
    # gray.shape[::-1] 是图像的(width, height)
    ret_cal, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret_cal:
        print("\n标定成功!")
        print("相机内参矩阵 (mtx):")
        print(mtx)
        print("\n畸变系数 (dist):")
        print(dist)
        
        # --- 5. 保存标定结果 ---
        # 使用numpy的.npz格式保存，方便读取
        np.savez('camera_calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print("\n标定结果已保存到 'camera_calibration_data.npz'")

        # 评估标定误差
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print(f"\n平均重投影误差: {mean_error/len(objpoints)}")
    else:
        print("标定失败。")