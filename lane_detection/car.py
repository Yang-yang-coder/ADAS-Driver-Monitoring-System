import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 解决matplotlib中文方框乱码
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# 以脚本所在目录为基准路径，确保无论从哪里运行都能找到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# 1. 预处理函数（返回灰度、模糊、边缘三张图）
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return gray, blur, edges

# 1b. 自适应Canny预处理（用于视频，适应帧间光照变化）
def preprocess_adaptive(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 根据图像中值自动计算Canny阈值
    sigma = 0.33
    median = np.median(blur)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(blur, lower, upper)
    return gray, blur, edges

# 2. ROI掩膜提取，屏蔽天空、建筑无关区域
def region_of_interest(edges):
    h, w = edges.shape
    mask = np.zeros_like(edges)
    # 梯形车道区域顶点
    poly = np.array([[(0, h), (w * 0.25, h * 0.6), (w * 0.75, h * 0.6), (w, h)]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    roi = cv2.bitwise_and(edges, mask)
    return roi

# 3. 绘制筛选后的车道线
def draw_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 斜率过滤水平杂线，避免路面横线、杂物干扰
            k = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(k) > 0.18:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # 原图与车道线融合
    return cv2.addWeighted(img, 0.8, line_img, 1, 0)

# 图片检测函数（你原来的功能完整保留）
def image_detect():
    # 读取道路图片
    img = cv2.imread("road5.png")
    # 图片读取容错判断
    if img is None:
        print("文件读取失败！请确认 road4.png 与当前py文件在同一个文件夹")
        exit()

    gray, blur, edges = preprocess(img)
    roi = region_of_interest(edges)
    # 概率霍夫变换检测直线段
    lines = cv2.HoughLinesP(roi, rho=2, theta=np.pi / 180, threshold=40,
                            minLineLength=15, maxLineGap=180)
    result = draw_lines(img, lines)

    # 2×2画布展示四步效果图（论文直接截图使用）
    plt.figure(figsize=(12, 9))
    plt.subplot(221), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("原图")
    plt.subplot(222), plt.imshow(edges, cmap="gray"), plt.title("Canny边缘检测图")
    plt.subplot(223), plt.imshow(roi, cmap="gray"), plt.title("ROI车道感兴趣区域")
    plt.subplot(224), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title("车道线检测最终结果")
    plt.tight_layout()
    plt.show()

    # 保存最终检测效果图到本地
    cv2.imwrite("lane_result.jpg", result)
    print("图片检测运行完成，结果已保存为 lane_result.jpg")

# 新增：视频检测函数，完全复用上面三个处理函数
def video_detect():
    # 参数说明：0=电脑摄像头；填写"video.mp4"读取本地视频文件
    cap = cv2.VideoCapture("road1.mp4")
    if not cap.isOpened():
        print("视频打开失败，请检查视频文件名与路径！")
        return

    # 获取视频基础属性（对应考核要求：读取视频文件属性）
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息：帧率={fps}, 分辨率={width}×{height}, 总帧数={total_frame}")

    # 创建视频写入对象，保存处理后的视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

    # 逐帧循环处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 复用和图片完全一样的处理逻辑
        gray, blur, edges = preprocess_adaptive(frame)
        roi = region_of_interest(edges)
        lines = cv2.HoughLinesP(roi, rho=2, theta=np.pi / 180, threshold=40,
                                minLineLength=15, maxLineGap=180)
        frame_result = draw_lines(frame, lines)
        # 写入输出视频
        out.write(frame_result)
        # 实时弹窗展示
        cv2.imshow("视频实时车道检测", frame_result)
        # 按q退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放所有资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("视频处理完成，已保存 output_video.mp4")

# 程序入口，可自由切换图片/视频模式
if __name__ == "__main__":
    # 运行图片检测
    image_detect()
    # 取消下面注释即可运行视频检测
    # video_detect()