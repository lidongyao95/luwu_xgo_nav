#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YDLidar Tmini Plus 配置与体检，在机器狗（树莓派 CM5）上运行。

用法：
    python3 lidar.py                             # 默认跑 6 秒
    python3 lidar.py --seconds 10 --out lidar_test.json

检查项：
    1. 能不能连上（/dev/ttyUSB0，230400）
    2. SDK 自检的健康状态（日志里的 "The health status good"）
    3. 帧率是否稳定在 10 Hz、每帧点数是否 410、有无丢帧
    4. 有效点数量、测距范围、强度是否像真实场景
    5. 存一帧完整扫描点（angle, range, intensity）到 JSON，供上位机画图

依赖：/opt/luwu-os/libs/ydlidar_sdk（狗上已装）。

注意：雷达转起来会一直转到 turnOff()；脚本退出前会关掉。
      同一时间只能有一个进程占用 /dev/ttyUSB0（webview.py 跑着时要先停）。
"""

import argparse
import json
import math
import sys
import time

LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = 230400
LIDAR_SDK = "/opt/luwu-os/libs/ydlidar_sdk"


def create_lidar():
    """实时采集与体检共用同一套设备选项。"""
    sys.path.insert(0, LIDAR_SDK)
    import ydlidar

    ydlidar.os_init()
    laser = ydlidar.CYdLidar()
    o = laser.setlidaropt
    # 由 SDK 自动识别 Tmini Plus 型号。
    o(ydlidar.LidarPropSerialPort, LIDAR_PORT)
    o(ydlidar.LidarPropSerialBaudrate, LIDAR_BAUD)
    o(ydlidar.LidarPropIntenstiy, True)
    o(ydlidar.LidarPropIntenstiyBit, 8)
    o(ydlidar.LidarPropFixedResolution, True)
    o(ydlidar.LidarPropReversion, False)
    o(ydlidar.LidarPropInverted, False)
    o(ydlidar.LidarPropAutoReconnect, True)
    o(ydlidar.LidarPropSingleChannel, False)
    o(ydlidar.LidarPropSupportMotorDtrCtrl, False)
    o(ydlidar.LidarPropSupportHeartBeat, False)
    o(ydlidar.LidarPropMaxAngle, 180.0)
    o(ydlidar.LidarPropMinAngle, -180.0)
    o(ydlidar.LidarPropMaxRange, 16.0)
    o(ydlidar.LidarPropMinRange, 0.05)
    o(ydlidar.LidarPropScanFrequency, 10.0)
    return ydlidar, laser


def main():
    ap = argparse.ArgumentParser(description="YDLidar Tmini Plus 体检")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", default="lidar_test.json")
    args = ap.parse_args()

    ydlidar, laser = create_lidar()
    t_init = time.time()
    if not laser.initialize():
        print("初始化失败：%s" % laser.DescribeError())
        sys.exit(3)
    t_ready = time.time()
    if not laser.turnOn():
        print("启动失败：%s" % laser.DescribeError())
        sys.exit(4)
    t_on = time.time()

    first = None
    scans = fails = 0
    samples, dump, cfg = [], None, None
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        scan = ydlidar.LaserScan()
        t_call = time.time()
        if not laser.doProcessSimple(scan):
            fails += 1
            time.sleep(0.005)
            continue
        if first is None:
            first = time.time() - t0
        scans += 1
        pts = list(scan.points)
        ang = [math.degrees(p.angle) for p in pts]
        rng = [p.range for p in pts]
        inten = [p.intensity for p in pts]
        valid = [r for r in rng if r > 0.03]
        samples.append({
            "t": round(time.time() - t0, 3), "n": len(pts), "nvalid": len(valid),
            "freq": round(scan.scanFreq, 2), "rate": round(scan.sampleRate, 1),
            "amin": round(min(ang), 1) if ang else None,
            "amax": round(max(ang), 1) if ang else None,
            "rmin": round(min(valid), 2) if valid else None,
            "rmax": round(max(valid), 2) if valid else None,
            "rmean": round(sum(valid) / len(valid), 2) if valid else None,
            "imax": round(max(inten), 0) if inten else None,
            "dt": round(time.time() - t_call, 4),
        })
        if cfg is None:
            cfg = {"min_angle": round(math.degrees(scan.config.min_angle), 2),
                   "max_angle": round(math.degrees(scan.config.max_angle), 2),
                   "angle_increment": round(math.degrees(scan.config.angle_increment), 3),
                   "min_range": round(scan.config.min_range, 3),
                   "max_range": round(scan.config.max_range, 3)}
        if dump is None and len(valid) > 250:
            dump = [(round(math.degrees(p.angle), 2), round(p.range, 3),
                     round(float(p.intensity), 1)) for p in pts if p.range > 0.03]

    t_end = time.time()
    try:
        laser.turnOff()
        laser.disconnect()
    except Exception:
        pass
    try:
        ydlidar.os_shutdown()
    except Exception:
        pass

    out = {"init_s": round(t_ready - t_init, 2), "turnon_s": round(t_on - t_ready, 2),
           "first_scan_s": round(first, 2) if first else None,
           "duration": round(t_end - t0, 2), "scans": scans, "fails": fails,
           "fps": round(scans / (t_end - t0), 2), "config": cfg,
           "samples": samples, "points": dump}
    with open(args.out, "w") as f:
        json.dump(out, f)

    print("初始化 %.2fs，启动扫描 %.2fs，首帧 %s s"
          % (out["init_s"], out["turnon_s"], out["first_scan_s"]))
    print("帧率 %.2f Hz（%d 帧 / %.1fs，失败 %d 次）" % (out["fps"], scans, out["duration"], fails))
    print("配置：%s" % cfg)
    if dump:
        print("整帧点数 %d，已存 %s" % (len(dump), args.out))
    else:
        print("警告：没有抓到有效整帧，检查雷达是否被挡住/供电")


if __name__ == "__main__":
    main()
