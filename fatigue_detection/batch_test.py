import cv2
import numpy as np
import os

# 复用 car.py 的处理函数
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return gray, blur, edges

def preprocess_adaptive(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sigma = 0.33
    median = np.median(blur)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(blur, lower, upper)
    return gray, blur, edges

def region_of_interest(edges):
    h, w = edges.shape
    mask = np.zeros_like(edges)
    poly = np.array([[(0, h), (w * 0.25, h * 0.6), (w * 0.75, h * 0.6), (w, h)]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    roi = cv2.bitwise_and(edges, mask)
    return roi

def draw_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            k = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(k) > 0.18:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return cv2.addWeighted(img, 0.8, line_img, 1, 0)

video_files = ["road.mp4", "road1.mp4", "road2.mp4", "road3.mp4", "road_video.mp4"]
os.makedirs("compare_frames", exist_ok=True)

results = []

for vf in video_files:
    if not os.path.exists(vf):
        print(f"⚠ 跳过，文件不存在: {vf}")
        continue

    cap = cv2.VideoCapture(vf)
    if not cap.isOpened():
        print(f"⚠ 无法打开: {vf}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_name = f"output_{vf}"
    out = cv2.VideoWriter(out_name, fourcc, fps, (width, height))

    frame_idx = 0
    total_lines = 0
    frames_with_lines = 0
    sample_saved = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray, blur, edges = preprocess_adaptive(frame)
        roi = region_of_interest(edges)
        lines = cv2.HoughLinesP(roi, rho=2, theta=np.pi/180, threshold=40,
                                minLineLength=15, maxLineGap=180)

        if lines is not None:
            total_lines += len(lines)
            frames_with_lines += 1

        result = draw_lines(frame, lines)
        out.write(result)

        # 保存第30帧作为样本对比
        if frame_idx == 30 and not sample_saved:
            cv2.imwrite(f"compare_frames/{vf}_frame30_roi.jpg", roi)
            cv2.imwrite(f"compare_frames/{vf}_frame30_result.jpg", result)
            sample_saved = True

        frame_idx += 1

    cap.release()
    out.release()

    line_ratio = frames_with_lines / max(frame_idx, 1) * 100
    avg_lines = total_lines / max(frames_with_lines, 1)
    results.append((vf, width, height, total, line_ratio, avg_lines, total_lines))
    print(f"[{vf}] {width}×{height}, {total}帧 | "
          f"检测到线的帧: {line_ratio:.0f}% | "
          f"平均每帧{avg_lines:.1f}条线 | "
          f"输出: {out_name}")

cv2.destroyAllWindows()

print("\n========== 对比总结 ==========")
print(f"{'视频文件':<18} {'分辨率':<14} {'总线数':<10} {'检出率':<10} {'平均线数'}")
print("-" * 65)
for vf, w, h, total, ratio, avg, tlines in results:
    print(f"{vf:<18} {w}×{h:<8} {tlines:<10} {ratio:.0f}%{'':<6} {avg:.1f}")

# 推荐最佳视频
if results:
    best = max(results, key=lambda x: x[4])  # 按检出率排序
    print(f"\n★ 检出率最高: {best[0]} ({best[4]:.0f}% 的帧检测到车道线)")
    best2 = max(results, key=lambda x: x[5])  # 按平均线数排序
    print(f"★ 平均线数最多: {best2[0]} (每帧 {best2[5]:.1f} 条)")

print(f"\n样本截图已保存到 compare_frames/ 目录")
