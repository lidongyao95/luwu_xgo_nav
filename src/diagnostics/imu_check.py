#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMU 体检 —— 在机器狗（树莓派 CM5）上运行

两个子命令：

    python3 imu_check.py static --seconds 30
        静止基线：采样 roll/pitch/yaw，输出噪声、漂移、是否"冻结"。
        判据：roll/pitch 静止 σ < 0.1° 为正常（实测约 0.02°）；
              yaw 静止完全不变是**固件死区**，正常但不可用于精细角度。

    python3 imu_check.py log --seconds 300 --out imu_move.csv
        连续记录 CSV，做手摆动态测试：抬前部、左右侧倾、绕竖直轴旋转。
        看的是"三轴是否各自独立响应、是否平滑、有无丢帧"。

依赖：xgolib（狗上已装，/usr/local/lib/python3.13/dist-packages）。

已知固件问题（M-6.0.2 与 M-7.0.0b8 都有）：
    dog.read_imu() 读的 0x65 寄存器没有正确填充，返回里混着固件版本字符串，
    会解出 2.99e36 这类垃圾值。判断 IMU 好坏**以 read_roll/pitch/yaw 为准**，
    需要角速度时用姿态角差分（onnx_runner.py 就是这么做的）。
"""

import argparse
import csv
import statistics as st
import time

from xgolib import XGO

PORT = "/dev/ttyAMA0"


def _open():
    dog = XGO(port=PORT, version="auto")
    print("firmware: %s    battery: %d%%" % (dog.read_firmware(), dog.read_battery()))
    return dog


def cmd_static(args):
    dog = _open()
    rolls, pitches, yaws, ts = [], [], [], []
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        rolls.append(dog.read_roll())
        pitches.append(dog.read_pitch())
        yaws.append(dog.read_yaw())
        ts.append(time.time() - t0)
        time.sleep(1.0 / args.hz)

    print("\n采样 %d 次 / %.1f s（%.1f Hz）" % (len(ts), ts[-1], len(ts) / ts[-1]))
    for name, v in (("roll", rolls), ("pitch", pitches), ("yaw", yaws)):
        print("%-5s mean %+7.2f  sd %6.3f  min %+7.2f  max %+7.2f  不同取值 %d 个"
              % (name, st.mean(v), st.pstdev(v), min(v), max(v), len(set(v))))
    print("yaw 漂移：%+.3f° / %.1fs" % (yaws[-1] - yaws[0], ts[-1]))

    verdict = []
    verdict.append("roll/pitch %s" % ("正常" if max(st.pstdev(rolls), st.pstdev(pitches)) < 0.1 else "噪声偏大"))
    verdict.append("yaw %s" % ("读数被固件锁住（静止不更新，属已知行为）" if len(set(yaws)) == 1 else "静止有正常抖动"))
    print("结论：" + "；".join(verdict))


def cmd_log(args):
    dog = _open()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "roll", "pitch", "yaw"])
        t0 = time.time()
        n = 0
        while time.time() - t0 < args.seconds:
            w.writerow(["%.3f" % (time.time() - t0),
                        "%.2f" % dog.read_roll(),
                        "%.2f" % dog.read_pitch(),
                        "%.2f" % dog.read_yaw()])
            f.flush()
            n += 1
            time.sleep(1.0 / args.hz)
    print("已写入 %s（%d 行 / %.0f s）" % (args.out, n, args.seconds))


def main():
    ap = argparse.ArgumentParser(description="XGO IMU 体检")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("static", help="静止基线检查")
    p1.add_argument("--seconds", type=float, default=30.0)
    p1.add_argument("--hz", type=float, default=16.0)
    p1.set_defaults(func=cmd_static)

    p2 = sub.add_parser("log", help="连续记录 CSV（做手摆动态测试）")
    p2.add_argument("--seconds", type=float, default=300.0)
    p2.add_argument("--hz", type=float, default=16.0)
    p2.add_argument("--out", default="imu_move.csv")
    p2.set_defaults(func=cmd_log)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
