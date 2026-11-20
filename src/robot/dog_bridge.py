#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGO 桥接服务 —— 把关节状态暴露成 HTTP，把本机算出来的目标角度写进舵机

设计要点：**每个控制周期只走一个 HTTP 往返**。
    GET /step?q=q1,q2,...,q12
        → 先下发这 12 个腿关节角，再回读 15 个关节角 + 姿态
        → 返回 {"q":[...], "roll":.., "pitch":.., "yaw":.., "t":..}

这样 Mac 上的策略循环是 40 Hz 时，每周期只需一次网络往返（而不是两次），
可以把 WiFi 延迟的影响压到最小。

    GET /state      只读，用于自检
    GET /stop       停车

用法（狗上）：
    sudo systemctl stop luwu-launcher        # 释放串口/相机（串口其实不冲突，但统一停掉更干净）
    python3 dog_bridge.py --port 8110
"""

import argparse
import math
import threading
import time

from flask import Flask, jsonify, request
from xgolib import XGO

LEG_SERVO_IDS = [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43]

app = Flask(__name__)
_lock = threading.Lock()
_dog = None


def dog():
    global _dog
    if _dog is None:
        _dog = XGO(port="/dev/ttyAMA0", version="auto")
        fw = _dog.read_firmware()
        if fw in ("Null", None, ""):
            raise RuntimeError("连不上机器狗")
        print("[bridge] 固件 %s" % fw, flush=True)
    return _dog


def read_state():
    d = dog()
    q = d.read_motor()
    if len(q) < 15:
        q = list(q) + [0.0] * (15 - len(q))
    return {
        "q": [round(float(v), 2) for v in q[:15]],
        "roll": round(float(d.read_roll()), 2),
        "pitch": round(float(d.read_pitch()), 2),
        "yaw": round(float(d.read_yaw()), 2),
        "t": round(time.time(), 4),
    }


@app.route("/state")
def state():
    with _lock:
        return jsonify(read_state())


@app.route("/step")
def step():
    """下发 12 个腿关节角，并回读状态。"""
    raw = request.args.get("q", "")
    vals = []
    if raw:
        try:
            vals = [float(x) for x in raw.split(",") if x != ""]
        except ValueError:
            return jsonify({"error": "bad q"}), 400
    # 安全钳位：任何关节目标都不允许超出 ±90°
    vals = [max(-90.0, min(90.0, v)) for v in vals[:12]]

    with _lock:
        d = dog()
        if len(vals) == 12:
            d.motor(LEG_SERVO_IDS, vals)
        return jsonify(read_state())


@app.route("/stop")
def stop():
    with _lock:
        try:
            d = dog()
            d.move_x(0)
            d.turn(0)
            d.stop()
        except Exception:
            pass
    return jsonify({"stopped": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8110)
    args = ap.parse_args()
    dog()                     # 预热，提前暴露串口问题
    print("[bridge] 监听 0.0.0.0:%d" % args.port, flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False,
            use_reloader=False)


if __name__ == "__main__":
    main()
