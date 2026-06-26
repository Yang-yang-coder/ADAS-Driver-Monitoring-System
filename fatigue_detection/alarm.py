"""报警模块
- 默认使用 pygame.mixer 播放生成的 WAV 警报音
- 若 pygame 不可用，回退到 Windows `winsound.Beep`
- 通过 cooldown 控制同一告警的播放频率，避免吵
"""
from __future__ import annotations

import math
import os
import struct
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional


def _synthesize_beep(path: str, freq: int = 880, duration_ms: int = 350, volume: float = 0.6) -> None:
    """合成一个简单的方波+正弦混合警报音。"""
    sample_rate = 44100
    n_samples = int(sample_rate * duration_ms / 1000)
    amplitude = int(32767 * max(0.0, min(1.0, volume)))

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            # 在高频上叠加一个低频，得到更"警笛"的听感
            v = 0.6 * math.sin(2 * math.pi * freq * t) + 0.4 * math.sin(2 * math.pi * (freq * 0.5) * t)
            sample = int(amplitude * v)
            frames.extend(struct.pack("<h", max(-32768, min(32767, sample))))
        wf.writeframes(bytes(frames))


def _ensure_wav(path: str) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    _synthesize_beep(path)
    return path


@dataclass
class _Channel:
    label: str
    cooldown: float        # 两次报警之间的最小间隔（秒）
    last_play: float = 0.0


class Alarm:
    """非阻塞警报播放器。"""

    def __init__(self,
                 alarm_wav: Optional[str] = None,
                 cooldown_fatigue: float = 1.0,
                 cooldown_yawn: float = 2.5,
                 cooldown_head: float = 3.0):
        self._lock = threading.Lock()
        self._enabled = True
        self._channels = {
            "fatigue": _Channel("fatigue", cooldown_fatigue),
            "yawn": _Channel("yawn", cooldown_yawn),
            "head": _Channel("head", cooldown_head),
        }

        if alarm_wav is None:
            alarm_dir = os.path.join(tempfile.gettempdir(), "fatigue_alarm")
            os.makedirs(alarm_dir, exist_ok=True)
            alarm_wav = os.path.join(alarm_dir, "alarm.wav")
        self._alarm_wav = _ensure_wav(alarm_wav)

        self._backend = "none"
        try:
            import pygame  # noqa: F401
            import pygame.mixer
            pygame.mixer.init()
            pygame.mixer.music.load(self._alarm_wav)
            self._backend = "pygame"
        except Exception:
            # Windows 自带 winsound
            try:
                import winsound  # noqa: F401
                self._backend = "winsound"
            except Exception:
                self._backend = "none"

    def enable(self, on: bool) -> None:
        self._enabled = on

    def trigger(self, kind: str) -> None:
        """kind: fatigue / yawn / head"""
        if not self._enabled:
            return
        ch = self._channels.get(kind)
        if ch is None:
            return
        now = time.time()
        with self._lock:
            if now - ch.last_play < ch.cooldown:
                return
            ch.last_play = now

        threading.Thread(target=self._play, args=(kind,), daemon=True).start()

    def _play(self, kind: str) -> None:
        try:
            if self._backend == "pygame":
                import pygame.mixer
                pygame.mixer.music.play()
            elif self._backend == "winsound":
                import winsound
                # 短促三连 beep
                for _ in range(3):
                    winsound.Beep(1000, 200)
                    time.sleep(0.05)
            # else: backend == "none" 静默
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._backend == "pygame":
                import pygame.mixer
                pygame.mixer.quit()
        except Exception:
            pass
