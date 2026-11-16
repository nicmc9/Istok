"""Генерирует тестовые звуки sandbox (синтез, без внешних файлов — лицензия CC0).

Запуск: python3 assets/audio/generate.py
"""
import math
import os
import random
import struct
import wave

HERE = os.path.dirname(os.path.abspath(__file__))


def write(name, rate, samples):
    peak = max(1e-9, max(abs(s) for s in samples))
    with wave.open(os.path.join(HERE, name), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"".join(struct.pack("<h", int(s / peak * 0.9 * 32767)) for s in samples))


def bounce():
    """Короткий глухой стук: затухающий тон с падающей частотой + щелчок шума."""
    rate, length = 44100, 0.22
    random.seed(1)
    out = []
    phase = 0.0
    for i in range(int(rate * length)):
        t = i / rate
        freq = 180 + 260 * math.exp(-t * 40)
        phase += 2 * math.pi * freq / rate
        tone = math.sin(phase) * math.exp(-t * 22)
        click = random.uniform(-1, 1) * math.exp(-t * 300) * 0.5
        out.append(tone + click)
    write("bounce.wav", rate, out)


def ambient():
    """Петля 8 с: мягкие аккорды (Am — F — C — G) синусами с медленной атакой.
    Начало и конец совпадают по фазе огибающей — шва при повторе не слышно."""
    rate, bar = 22050, 2.0
    chords = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (196.0, 261.63, 329.63), (196.0, 246.94, 293.66)]
    out = []
    for chord in chords:
        for i in range(int(rate * bar)):
            t = i / rate
            env = math.sin(math.pi * t / bar) ** 2  # плавно от 0 до 0 за такт
            s = sum(math.sin(2 * math.pi * f * t) + 0.3 * math.sin(4 * math.pi * f * t) for f in chord)
            out.append(s * env)
    write("ambient_loop.wav", rate, out)


bounce()
ambient()
