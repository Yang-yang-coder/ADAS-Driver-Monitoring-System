import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 解决matplotlib中文方框乱码
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# ==================== 工具函数 ====================
def imread_cn(path, flags=cv2.IMREAD_COLOR):
    """支持中文路径的图像读取（解决OpenCV Windows下中文路径问题）"""
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_cn(path, img):
    """支持中文路径的图像写入"""
    try:
        ext = os.path.splitext(path)[1]
        _, buf = cv2.imencode(ext, img)
        with open(path, "wb") as f:
            f.write(buf)
        return True
    except Exception:
        return False


def augment_face(img):
    """对单张灰度人脸图像做数据增强，返回增强后的图像列表"""
    augmented = [img]                     # 原图
    augmented.append(cv2.flip(img, 1))   # 水平翻转
    # 轻微亮度变化，模拟不同光照
    for delta in [-10, 10]:
        bright = cv2.add(img, delta)
        bright = np.clip(bright, 0, 255).astype(np.uint8)
        augmented.append(bright)
    return augmented


# ==================== 全局配置 ====================
# 人脸检测级联分类器路径（使用OpenCV自带的）
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# 训练数据存放目录
DATA_DIR = "face_data"
# 训练好的模型保存路径
MODEL_PATH = "face_model.yml"
# 姓名映射文件（label → 人名）
LABEL_MAP_PATH = "face_labels.txt"


# ==================== 1. 人脸检测 ====================
def detect_faces(img, scale_factor=1.1, min_neighbors=5, min_size=(80, 80)):
    """检测图像中的人脸，返回人脸矩形列表 [(x, y, w, h), ...]"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化，改善光照不均
    gray = cv2.equalizeHist(gray)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
        minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces, gray


def draw_face_boxes(img, faces, names=None, confidences=None):
    """在图像上绘制人脸框和标签"""
    img_copy = img.copy()
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = ""
        if names is not None and i < len(names):
            conf_str = ""
            if confidences is not None and i < len(confidences):
                conf_str = f" ({confidences[i]:.1f})"
            label = f"{names[i]}{conf_str}"
        if label:
            cv2.putText(img_copy, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img_copy


# ==================== 2. 人脸样本采集（摄像头） ====================
def collect_samples():
    """从摄像头采集人脸样本，按人名保存到 face_data/ 目录"""
    person_name = input("请输入采集对象姓名（英文或拼音，如 zhangsan）：").strip()
    if not person_name:
        print("姓名不能为空！")
        return

    save_dir = os.path.join(DATA_DIR, person_name)
    os.makedirs(save_dir, exist_ok=True)

    # 统计已有样本数，从已有最大编号+1继续
    existing = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
    count = len(existing)
    print(f"当前 {person_name} 已有 {count} 张样本，将从第 {count + 1} 张开始采集")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头打开失败！请检查摄像头是否可用")
        return

    print("\n========== 操作说明 ==========")
    print("  SPACE 键 → 拍摄一张样本")
    print("  ESC键  → 退出采集")
    print("  建议采集 20~30 张不同角度、不同表情的样本")
    print("==============================\n")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 镜像翻转，更自然
        faces, gray = detect_faces(frame, min_size=(60, 60))

        # 检测到人脸则画绿色框，否则画红色框提示
        if len(faces) == 1:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Ready - {person_name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif len(faces) > 1:
            cv2.putText(frame, "More than 1 face detected!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Count: {count}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Sample Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE键 — 保存当前人脸
            if len(faces) == 1:
                x, y, w, h = faces[0]
                # 稍微扩大人脸区域
                margin = 20
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                face_roi = frame[y1:y2, x1:x2]
                # 统一缩放到 200×200
                face_resized = cv2.resize(face_roi, (200, 200))
                # 转为灰度保存（减小文件体积，提升训练速度）
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                save_path = os.path.join(save_dir, f"{count + 1}.jpg")
                cv2.imwrite(save_path, face_gray)
                count += 1
                print(f"已保存第 {count} 张 → {save_path}")
            else:
                print("未检测到唯一人脸，请调整位置后重试")

        elif key == 27:  # ESC键 — 退出
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"采集完成！共为 {person_name} 采集了 {count} 张样本")


# ==================== 3. 模型训练 ====================
def train_model():
    """读取 face_data/ 下所有人脸样本，训练LBPH识别器并保存"""
    if not os.path.exists(DATA_DIR):
        print(f"数据目录 {DATA_DIR} 不存在！请先采集人脸样本")
        return

    # 获取所有人名（即子文件夹名）
    persons = [d for d in os.listdir(DATA_DIR)
               if os.path.isdir(os.path.join(DATA_DIR, d))]
    if len(persons) == 0:
        print(f"数据目录 {DATA_DIR} 下没有子文件夹！请先采集样本")
        return
    if len(persons) < 2:
        print(f"至少需要2个人的样本才能训练，当前仅检测到: {persons}")
        return

    print(f"检测到 {len(persons)} 个人: {persons}")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces_data = []
    labels = []
    label_map = {}  # label_id → 人名

    for label_id, person in enumerate(persons):
        person_dir = os.path.join(DATA_DIR, person)
        images = [f for f in os.listdir(person_dir) if f.endswith((".jpg", ".png", ".pgm"))]
        label_map[label_id] = person
        print(f"  {person}: {len(images)} 张样本")

        for img_name in images:
            img_path = os.path.join(person_dir, img_name)
            img = imread_cn(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # 尝试检测人脸并裁剪，提升训练与识别的一致性
            faces = face_cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) >= 1:
                # 检测到人脸 → 裁剪最大的一张用于训练
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                face_roi = img[y:y + h, x:x + w]
            else:
                # 未检测到人脸 → 可能是已裁剪样本，直接使用整张图
                face_roi = img

            face_resized = cv2.resize(face_roi, (200, 200))
            # 数据增强，扩充训练样本（原图 + 翻转 + 亮度变化 = 4倍）
            for aug_face in augment_face(face_resized):
                faces_data.append(aug_face)
                labels.append(label_id)

    if len(faces_data) == 0:
        print("未读取到有效样本！")
        return

    print(f"\n总训练样本数: {len(faces_data)}")
    print("正在训练 LBPH 识别器...")

    # 创建并训练LBPH人脸识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(faces_data, np.array(labels))

    # 保存模型
    recognizer.save(MODEL_PATH)
    print(f"模型已保存到 {MODEL_PATH}")

    # 保存姓名映射
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        for label_id, name in label_map.items():
            f.write(f"{label_id},{name}\n")
    print(f"姓名映射已保存到 {LABEL_MAP_PATH}")

    # 训练完成可视化：展示每个人的一张样本
    plt.figure(figsize=(4 * len(persons), 4))
    for label_id, person in enumerate(persons):
        person_dir = os.path.join(DATA_DIR, person)
        images = [f for f in os.listdir(person_dir) if f.endswith((".jpg", ".png"))]
        if images:
            sample = imread_cn(os.path.join(person_dir, images[0]), cv2.IMREAD_GRAYSCALE)
            if sample is not None:
                plt.subplot(1, len(persons), label_id + 1)
                plt.imshow(sample, cmap="gray")
                plt.title(f"{person} ({len(images)}张)")
                plt.axis("off")
    plt.suptitle("训练集概览", fontsize=16)
    plt.tight_layout()
    plt.show()

    print("训练完成！")


def load_model():
    """加载已训练的模型和姓名映射"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print("模型文件不存在！请先执行训练（模式2）")
        return None, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    label_map = {}
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                label_map[int(parts[0])] = parts[1]

    return recognizer, label_map


# ==================== 4. 实时人脸识别（摄像头） ====================
def realtime_recognition():
    """摄像头实时人脸识别，标注人名和置信度"""
    recognizer, label_map = load_model()
    if recognizer is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头打开失败！")
        return

    # 置信度阈值：低于此值认为是已知人脸，否则显示Unknown
    CONFIDENCE_THRESHOLD = 60.0

    print("\n========== 实时人脸识别 ==========")
    print("  ESC键 → 退出")
    print(f"  置信度阈值: {CONFIDENCE_THRESHOLD}")
    print("==================================\n")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        faces, gray = detect_faces(frame)

        names = []
        confidences = []

        for (x, y, w, h) in faces:
            # 提取人脸ROI并识别
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            label_id, confidence = recognizer.predict(face_roi)

            if confidence < CONFIDENCE_THRESHOLD:
                name = label_map.get(label_id, "Unknown")
            else:
                name = "Unknown"

            names.append(name)
            confidences.append(confidence)

        # 绘制结果
        result = draw_face_boxes(frame, faces, names, confidences)

        # 在画面顶部显示统计
        known_count = sum(1 for n in names if n != "Unknown")
        cv2.putText(result, f"Known: {known_count}  Unknown: {len(names) - known_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Real-time Face Recognition", result)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("实时识别已退出")


# ==================== 5. 图片人脸识别 ====================
def image_recognition(image_path=None):
    """对单张图片进行人脸检测与识别"""
    recognizer, label_map = load_model()
    if recognizer is None:
        return

    if image_path is None:
        image_path = input("请输入要识别的图片路径（如 people.png）：").strip()

    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return

    img = imread_cn(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    faces, gray = detect_faces(img)

    if len(faces) == 0:
        print("未检测到人脸！")
        # 仍展示原图
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("未检测到人脸")
        plt.axis("off")
        plt.show()
        return

    print(f"检测到 {len(faces)} 张人脸")

    names = []
    confidences = []

    for i, (x, y, w, h) in enumerate(faces):
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (200, 200))
        label_id, confidence = recognizer.predict(face_roi)

        CONFIDENCE_THRESHOLD = 60.0
        if confidence < CONFIDENCE_THRESHOLD:
            name = label_map.get(label_id, "Unknown")
        else:
            name = "Unknown"

        names.append(name)
        confidences.append(confidence)
        print(f"  人脸{i + 1}: {name}, 置信度={confidence:.1f}")

    # 绘制结果并展示
    result = draw_face_boxes(img, faces, names, confidences)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"人脸识别结果 — 检测到 {len(faces)} 张人脸")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 保存结果
    imwrite_cn("face_recognition_result.jpg", result)
    print("结果已保存为 face_recognition_result.jpg")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    print("=" * 45)
    print("     人脸识别系统 — Face Recognition")
    print("=" * 45)
    print("  1 — 采集人脸样本（摄像头）")
    print("  2 — 训练识别模型")
    print("  3 — 实时人脸识别（摄像头）")
    print("  4 — 图片人脸识别")
    print("  0 — 退出")
    print("-" * 45)

    try:
        choice = input("请选择模式 (0-4): ").strip()
    except EOFError:
        choice = "0"

    if choice == "1":
        collect_samples()
    elif choice == "2":
        train_model()
    elif choice == "3":
        realtime_recognition()
    elif choice == "4":
        image_recognition()
    elif choice == "0":
        print("已退出")
    else:
        print("无效选择！")
