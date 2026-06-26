"""疲劳驾驶检测 - 主程序
- 默认打开本机摄像头实时检测
- 也可指定 `--image` 跑静态图片（用于测试或截帧回放）
- 按 q 退出；按 m 切换静音；按 r 重置计数
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

from alarm import Alarm
from fatigue_detector import FatigueConfig, FatigueDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fatigue Driving Detector (face landmark based)")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--width", type=int, default=640, help="frame width")
    p.add_argument("--height", type=int, default=480, help="frame height")
    p.add_argument("--image", type=str, default=None, help="run on a single image and exit")
    p.add_argument("--video", type=str, default=None, help="run on a video file")
    p.add_argument("--no-sound", action="store_true", help="disable audio alarm")
    p.add_argument("--ear-thr", type=float, default=0.22, help="EAR threshold")
    p.add_argument("--mar-thr", type=float, default=0.65, help="MAR threshold")
    p.add_argument("--pitch-thr", type=float, default=20.0, help="pitch (head down) threshold (deg)")
    return p.parse_args()


def run_camera(args) -> int:
    cfg = FatigueConfig(
        ear_threshold=args.ear_thr,
        mar_threshold=args.mar_thr,
        pitch_down_threshold=args.pitch_thr,
    )
    detector = FatigueDetector(cfg)
    alarm = Alarm()
    alarm.enable(not args.no_sound)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera index", args.camera, file=sys.stderr)
        return 1

    print("[INFO] Press 'q' quit, 'm' mute toggle, 'r' reset counters")
    muted = args.no_sound
    fps_ts = time.time()
    fps_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame grab failed, exiting")
                break

            out = detector.process(frame)
            s = detector.state

            # 报警
            if not muted:
                if s.fatigue:
                    alarm.trigger("fatigue")
                if s.yawning:
                    alarm.trigger("yawn")
                if s.head_down:
                    alarm.trigger("head")

            # FPS
            fps_count += 1
            if time.time() - fps_ts >= 1.0:
                s.fps = fps_count / (time.time() - fps_ts)
                fps_count = 0
                fps_ts = time.time()
            cv2.putText(out, f"FPS: {s.fps:4.1f}", (out.shape[1] - 140, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(out, f"Sound: {'OFF' if muted else 'ON'}  (m to toggle)",
                        (out.shape[1] - 360, out.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Fatigue Driving Detector", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                muted = not muted
                alarm.enable(not muted)
                print(f"[INFO] Sound {'OFF' if muted else 'ON'}")
            if key == ord("r"):
                detector.state.blink_count = 0
                detector.state.yawn_count = 0
                print("[INFO] Counters reset")
    finally:
        cap.release()
        detector.close()
        alarm.close()
        cv2.destroyAllWindows()
    return 0


def run_image(args) -> int:
    cfg = FatigueConfig(
        ear_threshold=args.ear_thr,
        mar_threshold=args.mar_thr,
        pitch_down_threshold=args.pitch_thr,
    )
    detector = FatigueDetector(cfg)
    frame = cv2.imread(args.image)
    if frame is None:
        print("[ERROR] Cannot read image:", args.image, file=sys.stderr)
        return 1
    out = detector.process(frame)
    s = detector.state
    print(f"EAR={s.ear:.3f}  MAR={s.mar:.3f}  "
          f"Pitch={s.pitch:.1f}  Yaw={s.yaw:.1f}  Roll={s.roll:.1f}")
    print(f"Fatigue={s.fatigue}  Yawning={s.yawning}  HeadDown={s.head_down}")
    cv2.imshow("Fatigue Driving Detector", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    detector.close()
    return 0


def run_video(args) -> int:
    cfg = FatigueConfig(
        ear_threshold=args.ear_thr,
        mar_threshold=args.mar_thr,
        pitch_down_threshold=args.pitch_thr,
    )
    detector = FatigueDetector(cfg)
    alarm = Alarm()
    alarm.enable(not args.no_sound)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERROR] Cannot open video:", args.video, file=sys.stderr)
        return 1
    print("[INFO] Press 'q' quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = detector.process(frame)
            s = detector.state
            if not args.no_sound:
                if s.fatigue:
                    alarm.trigger("fatigue")
                if s.yawning:
                    alarm.trigger("yawn")
                if s.head_down:
                    alarm.trigger("head")
            cv2.imshow("Fatigue Driving Detector", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        detector.close()
        alarm.close()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    if args.image:
        return run_image(args)
    if args.video:
        return run_video(args)
    return run_camera(args)


if __name__ == "__main__":
    sys.exit(main())
