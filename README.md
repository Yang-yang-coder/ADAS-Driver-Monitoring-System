# ADAS-Driver-Monitoring-System
ADAS Driver Monitoring System Based on OpenCV and MediaPipe

基于 Python、OpenCV、MediaPipe 开发的驾驶员监控系统，实现车道线检测、疲劳驾驶检测及驾驶员身份识别等功能，为智能驾驶辅助系统提供基础视觉感知能力。

📖 项目简介：
本项目采用计算机视觉技术构建驾驶员监控系统（ADAS），主要包含车道线检测、疲劳驾驶检测和人脸识别三个功能模块。
项目通过 OpenCV 完成道路图像处理与车道线识别，结合 MediaPipe Face Mesh 提取人脸关键点，实现驾驶员疲劳状态分析，并利用 LBPH 人脸识别算法完成驾驶员身份识别，可支持图片、视频及摄像头实时检测。

✨ 项目特色：
🛣️ 车道线检测：基于 Canny 边缘检测、ROI 感兴趣区域提取及霍夫直线变换实现道路车道线检测。
😴 疲劳驾驶检测：基于 MediaPipe Face Mesh 提取 468 个人脸关键点，结合 EAR、MAR、PERCLOS 等指标综合判断驾驶员疲劳状态。
👤 人脸识别：采用 OpenCV LBPH 人脸识别算法，实现驾驶员身份识别及日志记录。
🎥 多场景支持：支持图片检测、视频检测及摄像头实时检测。

🛠 技术栈：Python、、OpenCV、MediaPipe、NumPy、Pillow、LBPH

📂 项目结构：
ADAS-Driver-Monitoring-System
│
├── lane_detection/          # 车道线检测模块
├── fatigue_detection/       # 疲劳驾驶检测模块
├── face_recognition/        # 人脸识别模块
├── screenshots/             # 项目截图
├── docs/                    # 项目文档
├── README.md

📸 项目展示：
🛣️ 车道线检测：展示车辆行驶过程中车道线检测效果。
![Lane Detection](screenshots/Figure_1.png)

😴 疲劳驾驶检测：实时检测驾驶员闭眼、打哈欠及头部姿态变化。
![Fatigue Detection](screenshots/fatigue_detection.png)

3️⃣ 运行疲劳驾驶检测：
python fatigue_detection/main.py

📌 后续优化：
将车道线检测、疲劳驾驶检测及人脸识别整合为统一的 ADAS 系统界面。
增加车道偏离预警（LDW）功能。
提升复杂光照及夜间环境下的检测稳定性。
优化实时检测性能，提高系统运行效率。

👨‍💻 作者：杨扬
Yang Yang
软件工程专业 · Python 开发 · 计算机视觉方向

⭐ 如果这个项目对你有所帮助，欢迎 Star 本仓库。
