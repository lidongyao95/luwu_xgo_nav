#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""录包 —— 在狗上运行，把雷达原始帧 + IMU 全速率采样写成 JSONL

设计原则：**录制只管采集，不做任何处理**。SLAM、滤波、回放都在离线做，
这样同一份包可以反复跑不同算法，硬件问题也不会被算法掩盖。

用法（狗上）：
    python3 record_bag.py --seconds 60 -o bag_20261120.jsonl
    python3 record_bag.py -o walk1.jsonl          # Ctrl+C 结束

数据格式（每行一个 JSON）：
    {"type":"meta","t":<开始时间>,"firmware":"M-7.0.0b8","note":"..."}
    {"type":"imu","t":<狗端秒>,"r":..,"p":..,"y":..}
    {"type":"scan","t":<狗端秒>,"id":<帧号>,"pts":[[deg,m],...]}

为什么在狗上录：连本机 127.0.0.1，不经过 WiFi，帧和 IMU 都不会因网络抖动
丢序或重复；`since` 游标保证不重复也不漏（webview 端各留了 200 帧 / 600 个
IMU 样本的缓冲）。
"""

import argparse
import json
import sys
import time

import requests

BASE = "http://127.0.0.1:8000"


def main():
    ap = argparse.ArgumentParser(description="XGO 录包（雷达 + IMU）")
    ap.add_argument("-o", "--out", default="bag.jsonl")
    ap.add_argument("--seconds", type=float, default=0, help="0 = 一直录到 Ctrl+C")
    ap.add_argument("--base", default=BASE, help="webview 地址（默认本机）")
    args = ap.parse_args()

    # 先问一次当前游标，避免把缓冲里的陈旧数据当成新包
    try:
        f0 = requests.get(args.base + "/frame_id", timeout=3).json()
        last_id = int(f0.get("id", -1))
        last_t = time.time()
    except Exception as e:
        print("连不上 webview：%s" % e)
        sys.exit(2)

    out = open(args.out, "w")
    out.write(json.dumps({"type": "meta", "t": round(last_t, 3),
                          "note": "xgo bag: raw scan + imu"}) + "\n")
    out.flush()

    n_scan = n_imu = 0
    t0 = time.time()
    last_report = t0
    try:
        while True:
            if args.seconds and time.time() - t0 > args.seconds:
                break
            try:
                d = requests.get(args.base + "/frames",
                                 params={"since": last_id, "max": 30},
                                 timeout=2).json()
                for f in d.get("frames", []):
                    if f["id"] <= last_id:
                        continue
                    last_id = f["id"]
                    out.write(json.dumps({"type": "scan", "t": round(f["t"], 3),
                                          "id": f["id"], "pts": f["pts"]}) + "\n")
                    n_scan += 1
                im = requests.get(args.base + "/imu",
                                  params={"since": last_t, "max": 200},
                                  timeout=2).json()
                for s in im.get("imu", []):
                    if s[0] <= last_t:
                        continue
                    last_t = s[0]
                    out.write(json.dumps({"type": "imu", "t": round(s[0], 3),
                                          "r": round(s[1], 2), "p": round(s[2], 2),
                                          "y": round(s[3], 2)}) + "\n")
                    n_imu += 1
            except Exception as e:
                print("[warn] 读取失败：%s" % e)
                time.sleep(0.2)
                continue

            out.flush()
            now = time.time()
            if now - last_report >= 1.0:
                print("\r[rec] %.0fs  雷达 %d 帧  IMU %d 样本" % (now - t0, n_scan, n_imu),
                      end="", flush=True)
                last_report = now
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
    dur = time.time() - t0
    print("\n[rec] 完成：%.1f s，雷达 %d 帧（%.1f Hz），IMU %d 样本（%.1f Hz）→ %s"
          % (dur, n_scan, n_scan / max(dur, 1e-6), n_imu, n_imu / max(dur, 1e-6), args.out))


if __name__ == "__main__":
    main()
