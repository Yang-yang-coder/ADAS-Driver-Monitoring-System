"""疲劳驾驶检测核心模块
- 基于 MediaPipe Face Mesh 的 468 关键点
- 计算 EAR (Eye Aspect Ratio) 判定闭眼/眨眼
- 计算 MAR (Mouth Aspect Ratio) 判定打哈欠
- 通过 PnP 解算头部姿态 (yaw / pitch / roll) 判定低头/偏头
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe Face Mesh 关键点索引
# 眼睛（6 点法）
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# 嘴（外轮廓 4 点 + 内轮廓 4 点）
MOUTH_OUTER = [61, 291, 0, 17]   # 左 / 右 / 上 / 下
MOUTH_INNER = [13, 14, 78, 308]
# 头部姿态关键点
POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 199,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# 3D 模型点（与 POSE_LANDMARKS 顺序一致），单位：mm，通用人脸模型
MODEL_3D = np.array([
    (0.0, 0.0, 0.0),            # nose tip
    (0.0, -63.6, -12.5),        # chin
    (-43.3, 32.7, -26.0),       # left eye outer
    (43.3, 32.7, -26.0),        # right eye outer
    (-28.9, -28.9, -24.1),      # left mouth
    (28.9, -28.9, -24.1),       # right mouth
], dtype=np.float64)


def _euclid(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def eye_aspect_ratio(eye_points) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)"""
    p1, p2, p3, p4, p5, p6 = eye_points
    return (_euclid(p2, p6) + _euclid(p3, p5)) / (2.0 * _euclid(p1, p4) + 1e-6)


def mouth_aspect_ratio(mouth_points) -> float:
    """MAR = vertical / horizontal"""
    left, right, top, bottom = mouth_points
    vertical = (_euclid(top, bottom)) / 2.0
    horizontal = _euclid(left, right) + 1e-6
    return vertical / horizontal


@dataclass
class FatigueConfig:
    ear_threshold: float = 0.22         # 低于此值认为闭眼
    ear_consec_frames: int = 20         # 连续多少帧闭眼触发疲劳
    mar_threshold: float = 0.65         # 高于此值认为打哈欠
    mar_consec_frames: int = 15         # 连续多少帧张嘴触发打哈欠
    pitch_down_threshold: float = 20.0  # 低头角度（度）
    pitch_down_consec_frames: int = 30  # 连续多少帧低头
    perclos_window: float = 30.0        # PERCLOS 统计窗口（秒）


@dataclass
class FatigueState:
    ear_counter: int = 0
    ear_total: int = 0
    mar_counter: int = 0
    mar_total: int = 0
    pitch_counter: int = 0
    pitch_total: int = 0
    blink_count: int = 0
    yawn_count: int = 0
    perclos: float = 0.0                # Percentage of Eye Closure
    fatigue: bool = False
    yawning: bool = False
    head_down: bool = False
    fps: float = 0.0
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    # 内部缓存
    _ear_history: deque = field(default_factory=lambda: deque(maxlen=300))
    _closed_flags: deque = field(default_factory=lambda: deque(maxlen=900))
    _t0: float = field(default_factory=time.time)
    _last_ts: float = field(default_factory=time.time)


class FatigueDetector:
    """单例风格检测器；内部复用 MediaPipe 的 FaceMesh。"""

    def __init__(self, config: Optional[FatigueConfig] = None):
        self.config = config or FatigueConfig()
        self.state = FatigueState()
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        try:
            self._mesh.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _landmark_to_point(landmark, w: int, h: int) -> tuple[int, int]:
        return int(landmark.x * w), int(landmark.y * h)

    @staticmethod
    def _get_points(landmarks, indices, w: int, h: int) -> list[tuple[int, int]]:
        return [FatigueDetector._landmark_to_point(landmarks[i], w, h) for i in indices]

    def _estimate_head_pose(self, landmarks, w: int, h: int):
        image_pts = np.array([
            self._landmark_to_point(landmarks[idx], w, h)
            for idx in POSE_LANDMARKS.values()
        ], dtype=np.float64)

        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist = np.zeros((4, 1))

        ok, rvec, _ = cv2.solvePnP(
            MODEL_3D, image_pts, camera_matrix, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        rmat, _ = cv2.Rodrigues(rvec)
        # 欧拉角：pitch(x) yaw(y) roll(z)
        sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
        singular = sy < 1e-6
        if not singular:
            pitch = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
            yaw = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
            roll = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
        else:
            pitch = float(np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1])))
            yaw = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
            roll = 0.0
        return pitch, yaw, roll

    # ------------------------------------------------------------------ main
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """处理一帧 BGR 图像，返回绘制后的 BGR 图像。"""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)

        s = self.state
        cfg = self.config
        s.fatigue = s.yawning = s.head_down = False
        s.ear = s.mar = s.pitch = s.yaw = s.roll = 0.0

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0].landmark

            left_eye = self._get_points(face, LEFT_EYE, w, h)
            right_eye = self._get_points(face, RIGHT_EYE, w, h)
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            s.ear = ear

            mouth_outer = self._get_points(face, MOUTH_OUTER, w, h)
            mar = mouth_aspect_ratio(mouth_outer)
            s.mar = mar

            pose = self._estimate_head_pose(face, w, h)
            if pose is not None:
                s.pitch, s.yaw, s.roll = pose

            # ---- EAR 闭眼逻辑 ----
            if ear < cfg.ear_threshold:
                s.ear_counter += 1
                s.ear_total += 1
            else:
                if 3 <= s.ear_counter <= cfg.ear_consec_frames:
                    s.blink_count += 1
                s.ear_counter = 0

            # ---- MAR 打哈欠逻辑 ----
            if mar > cfg.mar_threshold:
                s.mar_counter += 1
                s.mar_total += 1
            else:
                if s.mar_counter >= cfg.mar_consec_frames:
                    s.yawn_count += 1
                s.mar_counter = 0

            # ---- Pitch 低头逻辑 ----
            if s.pitch > cfg.pitch_down_threshold:
                s.pitch_counter += 1
                s.pitch_total += 1
            else:
                s.pitch_counter = 0

            # ---- 触发判定 ----
            if s.ear_counter >= cfg.ear_consec_frames:
                s.fatigue = True
            if s.mar_counter >= cfg.mar_consec_frames:
                s.yawning = True
            if s.pitch_counter >= cfg.pitch_down_consec_frames:
                s.head_down = True

            # ---- PERCLOS（近 N 秒窗口）----
            now = time.time()
            s._closed_flags.append((now, ear < cfg.ear_threshold))
            window_start = now - cfg.perclos_window
            while s._closed_flags and s._closed_flags[0][0] < window_start:
                s._closed_flags.popleft()
            if s._closed_flags:
                s.perclos = sum(f for _, f in s._closed_flags) / len(s._closed_flags) * 100.0

            # ---- 可视化 ----
            self._draw_eye(frame_bgr, left_eye)
            self._draw_eye(frame_bgr, right_eye)
            self._draw_mouth(frame_bgr, mouth_outer)
            for idx in POSE_LANDMARKS.values():
                x, y = self._landmark_to_point(face[idx], w, h)
                cv2.circle(frame_bgr, (x, y), 2, (0, 255, 255), -1)
        else:
            # 没有检测到人脸，重置短计数器
            s.ear_counter = 0
            s.mar_counter = 0
            s.pitch_counter = 0

        # ---- 面板 ----
        self._draw_hud(frame_bgr)
        return frame_bgr

    # ------------------------------------------------------------------ draw
    @staticmethod
    def _draw_eye(img, pts):
        for i in range(len(pts)):
            cv2.line(img, pts[i], pts[(i + 1) % len(pts)], (0, 255, 0), 1)

    @staticmethod
    def _draw_mouth(img, pts):
        for i in range(len(pts)):
            cv2.line(img, pts[i], pts[(i + 1) % len(pts)], (255, 0, 0), 1)

    def _draw_hud(self, img: np.ndarray) -> None:
        s = self.state
        cfg = self.config
        h, w = img.shape[:2]
        panel_w = 320
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, 230), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        lines = [
            f"EAR : {s.ear:.3f}   thr {cfg.ear_threshold:.2f}",
            f"MAR : {s.mar:.3f}   thr {cfg.mar_threshold:.2f}",
            f"Pitch/Yaw/Roll : {s.pitch:5.1f} / {s.yaw:5.1f} / {s.roll:5.1f}",
            f"Blink : {s.blink_count}    Yawn : {s.yawn_count}",
            f"PERCLOS(30s) : {s.perclos:5.1f}%",
            f"EAR close counter : {s.ear_counter}",
            f"MAR open  counter : {s.mar_counter}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(img, text, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # 报警条
        if s.fatigue:
            self._draw_alert(img, "FATIGUE - WAKE UP!", (0, 0, 255))
        elif s.yawning:
            self._draw_alert(img, "YAWN DETECTED", (0, 165, 255))
        elif s.head_down:
            self._draw_alert(img, "HEAD DOWN - PAY ATTENTION", (0, 255, 255))

    @staticmethod
    def _draw_alert(img, text: str, color):
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h - 60), (w, h), color, -1)
        cv2.putText(img, text, (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
