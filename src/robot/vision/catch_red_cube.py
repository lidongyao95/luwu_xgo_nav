#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGO 机器狗 —— 红色方块视觉识别与抓取

和官方/教学代码里的「蓝色方块」「红色小球」版本相比，这个版本做了几处关键修正：

1. 相机通道：Picamera2 用 RGB888 配置时，capture_array() 实际给的是 **BGR** 顺序。
   教学代码里再套一次 cv2.COLOR_RGB2BGR 会让画面整体偏蓝、红色被压掉，
   本脚本直接按 BGR 处理，不做多余转换。
2. 方块识别：方块是直角多边形，用「轮廓 + 最小外接矩形 + 矩形度」判断，
   而不是霍夫圆；同时用边长反推距离（边长 ∝ 1/距离）。
3. 距离系数 K 可标定：先跑 calib 子命令量出 K，抓取距离就准了。
4. 默认不动手：不带 --grab 只做识别并保存标注图，确认识别稳定再开抓取。

用法
----
    # 1) 只识别，不动狗（安全，先跑这个调阈值）
    python3 catch_red_cube.py vision --frames 40

    # 2) 标定距离系数：把方块摆在 30cm 处，量出画面里的边长，自动算 K
    python3 catch_red_cube.py calib --dist 30

    # 3) 正式抓取
    python3 catch_red_cube.py grab

注意：luwu_launcher 的预览进程会独占摄像头，跑之前先
    sudo systemctl stop luwu-launcher
跑完再 sudo systemctl start luwu-launcher
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------- 可调参数 --

CAM_W, CAM_H = 640, 480          # 采集分辨率
PANEL_CX = CAM_W // 2            # 画面中心

# 红色在 HSV 里跨越 0/180 边界，必须两段一起取
RED_RANGES = (
    ((0,   100, 55), (12,  255, 255)),
    ((168, 100, 55), (180, 255, 255)),
)

MIN_AREA_PX = 400        # 最小轮廓面积（像素²），滤掉噪点
MAX_AREA_RATIO = 0.22    # 面积上限（占整幅比例），滤掉显示器/墙面等大面积红色
RECT_MIN = 0.72          # 矩形度下限：轮廓面积 / 最小外接矩形面积
ASPECT_MIN, ASPECT_MAX = 0.62, 1.60   # 方块面对角/旋转后长宽比范围
SOLIDITY_MIN = 0.85      # 面积 / 凸包面积，滤掉细长红色物体
EDGE_MARGIN = 8          # 框线贴边多少像素内算「被画面截断」，直接丢弃
MIN_CY_RATIO = 0.40      # 方块中心最低要在画面这个比例高度以下（桌面目标）

CUBE_K = 900.0           # 距离系数：dist_cm = K / 边长px（用 calib 标定）
NEAR_CM, FAR_CM = 13.0, 17.0          # 抓取窗口，位于其中才动手
CENTER_TOL_PX = 18       # 水平居中容差

ARM_LIFT_X, ARM_LIFT_Z = 60, 80       # 提臂安全位
LIE_Z, LIE_P = 75, 30                 # 趴下姿态

MAX_STEPS = 40           # 主循环最大步数，防跑飞
STEP_PAUSE = 0.35        # 每步之间的停顿，等姿态稳定


# ---------------------------------------------------------------- 工具函数 --

def hsv_mask(frame, ranges=RED_RANGES):
    """按 HSV 双区间提取红色掩膜，并做开闭运算去噪/补洞。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_cubes(frame, min_area=MIN_AREA_PX):
    """找出画面里所有像「红色方块」的目标，按可信度排序返回。"""
    h, w = frame.shape[:2]
    max_area = MAX_AREA_RATIO * h * w
    mask = hsv_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cubes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        if rw < 1 or rh < 1:
            continue

        # 桌面上的方块应当完整出现在画面里；贴边的多半是显示器/墙面/桌沿
        x0, y0, bw, bh = cv2.boundingRect(cnt)
        if (x0 <= EDGE_MARGIN or y0 <= EDGE_MARGIN
                or x0 + bw >= w - EDGE_MARGIN or y0 + bh >= h - EDGE_MARGIN):
            continue

        # 要抓的方块在桌面上、位于画面下半部，太靠上的不抓
        if cy < h * MIN_CY_RATIO:
            continue

        rect_area = rw * rh
        rectangularity = area / rect_area
        if rectangularity < RECT_MIN:
            continue

        aspect = min(rw, rh) / max(rw, rh)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area <= 0 or area / hull_area < SOLIDITY_MIN:
            continue

        side = (rw + rh) / 2.0
        cubes.append({
            "cx": int(cx),
            "cy": int(cy),
            "side": float(side),
            "area": float(area),
            "angle": float(angle),
            "rect": rect,
            "box": cv2.boxPoints(rect).astype(np.int32),
            "score": rectangularity * aspect,
        })

    # 抓取目标是「正前方桌面上」的方块：优先画面下方、靠近中线的
    cubes.sort(key=lambda c: (-c["area"], abs(c["cx"] - PANEL_CX)))
    return cubes


def distance_cm(side_px, k=CUBE_K):
    """由方块边长估算距离（cm）。边长与距离成反比。"""
    if side_px <= 0:
        return float("inf")
    return k / float(side_px)


def pick_target(cubes):
    """在候选里挑一个最适合抓的：面积够大且在画面下半部。"""
    if not cubes:
        return None
    lower_half = [c for c in cubes if c["cy"] > CAM_H * 0.35]
    pool = lower_half or cubes
    # 面积最大优先（离得最近），其次居中
    return max(pool, key=lambda c: (c["area"], -abs(c["cx"] - PANEL_CX)))


def annotate(frame, cube, dist=None):
    """把检测结果画到图上，用于人工核对。"""
    vis = frame.copy()
    vis = cv2.resize(vis, (CAM_W, CAM_H))
    cv2.line(vis, (PANEL_CX, 0), (PANEL_CX, CAM_H - 1), (0, 255, 255), 1)
    if cube is None:
        cv2.putText(vis, "NO RED CUBE", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis

    cv2.drawContours(vis, [cube["box"]], -1, (0, 255, 0), 2)
    cv2.circle(vis, (cube["cx"], cube["cy"]), 3, (0, 255, 0), -1)
    txt = "cx=%d cy=%d side=%.0fpx" % (cube["cx"], cube["cy"], cube["side"])
    if dist is not None:
        txt += " d=%.1fcm" % dist
    cv2.putText(vis, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis


class Camera:
    """最小的 Picamera2 封装，返回 BGR 帧。"""

    def __init__(self, width=CAM_W, height=CAM_H):
        from picamera2 import Picamera2
        self.picam = Picamera2()
        cfg = self.picam.create_preview_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        self.picam.configure(cfg)
        self.picam.start()
        time.sleep(0.6)          # 等 AE/AWB 收敛
        for _ in range(5):       # 丢掉前几帧，避免首帧发黑
            self.picam.capture_array()

    def read(self):
        # 注意：这里拿到的数组已经是 BGR，不要再做 RGB2BGR
        return self.picam.capture_array()

    def close(self):
        try:
            self.picam.stop()
            self.picam.close()
        except Exception:
            pass


# ---------------------------------------------------------------- 机器狗动作 --

def dog_connect():
    from xgolib import XGO
    # version="auto" 会读固件首字母自动选机型（本机是 M-6.0.2 → xgomini）。
    # 教学代码里硬编码 "xgolite" 会用错参数表，导致平移/姿态限位被裁错。
    dog = XGO(port="/dev/ttyAMA0", version="auto")
    fw = dog.read_firmware()
    if fw in ("Null", None, ""):
        raise RuntimeError("连不上机器狗（串口无应答），检查 /dev/ttyAMA0 与狗是否上电")
    print("[dog] 固件版本：%s" % fw)
    return dog


def lie_down(dog):
    """低姿态：压低身体 + 低头，让相机能贴近桌面看前方。"""
    dog.translation("z", LIE_Z)
    dog.attitude("p", LIE_P)
    time.sleep(1.2)


def grasp(dog, claw_close=200):
    """标准抓取动作序列：开爪 → 伸到最低 → 下降 → 合爪 → 抬起。"""
    print("[grasp] 开爪并伸向目标...")
    dog.arm_mode(0)
    time.sleep(0.2)
    dog.claw(0)
    time.sleep(0.3)

    dog.arm(120, -5)
    time.sleep(1.6)
    dog.arm(145, -37)        # 末端到桌面高度
    time.sleep(0.9)

    print("[grasp] 合爪")
    dog.claw(claw_close)
    time.sleep(1.4)

    print("[grasp] 抬起")
    dog.arm(ARM_LIFT_X, ARM_LIFT_Z)
    time.sleep(1.0)
    dog.attitude("p", 10)
    time.sleep(0.5)


def step_towards(dog, cube, dist, cfg):
    """根据当前检测结果走一步，返回 True 表示已就位可抓。

    顺序很重要：**先修前后距离，再在抓取窗口内做横向对准**。
    XGO 的横移步态会带着身体往前漂（实测 20 次 move_y 让距离从 15cm
    掉到 11cm），所以横移只在已经进入窗口后才做，漂出去就重新修距离。
    """
    x_err = cube["cx"] - PANEL_CX

    # 1) 前后：先把距离拉进窗口
    if dist > cfg.far:
        vx = 7 if dist > cfg.far + 6 else 4
        print("[approach] 距离 %.1fcm 偏远 → move_x(%d)" % (dist, vx))
        dog.move_x(vx)
        time.sleep(cfg.step_pause)
        dog.stop()
        return False

    if dist < cfg.near:
        print("[approach] 距离 %.1fcm 偏近 → move_x(-5)" % dist)
        dog.move_x(-5)
        time.sleep(cfg.step_pause)
        dog.stop()
        return False

    # 2) 窗口内做横向对准（固定步长，和他们验证过的写法一致）
    if abs(x_err) > cfg.center_tol:
        dy = 5 if x_err < 0 else -5
        print("[align] 距离 %.1fcm 已就位，横向偏 %+dpx → move_y(%+d)"
              % (dist, x_err, dy))
        dog.move_y(dy)
        time.sleep(cfg.step_pause)
        dog.stop()
        return False

    print("[ready] 距离 %.1fcm、水平偏差 %+dpx，可以抓了" % (dist, x_err))
    return True


def arm_preview(dog, cam, save_path):
    """只伸臂到抓取位、不合爪，拍一张照片看夹爪对不对得上方块。"""
    print("[preview] 伸臂到抓取位（不合爪）...")
    dog.arm_mode(0)
    time.sleep(0.2)
    dog.claw(0)
    time.sleep(0.3)
    dog.arm(120, -5)
    time.sleep(1.6)
    dog.arm(145, -37)
    time.sleep(1.0)

    frame = cam.read()
    cube = pick_target(find_cubes(frame))
    cv2.imwrite(save_path, annotate(frame, cube))
    print("[preview] 已存 %s" % save_path)

    dog.arm(ARM_LIFT_X, ARM_LIFT_Z)
    time.sleep(1.0)


# ---------------------------------------------------------------- 各子命令 --

def cmd_vision(args):
    """只识别，不驱动任何关节。用来调 HSV / 确认能稳定看到方块。"""
    cam = Camera()
    out_dir = args.save_dir
    os.makedirs(out_dir, exist_ok=True)

    hits = 0
    for i in range(args.frames):
        frame = cam.read()
        cube = pick_target(find_cubes(frame, args.min_area))
        dist = distance_cm(cube["side"], args.k) if cube else None
        if cube:
            hits += 1
            print("frame %02d: cx=%d cy=%d side=%.0fpx d=%.1fcm area=%.0f"
                  % (i, cube["cx"], cube["cy"], cube["side"], dist, cube["area"]))
        else:
            print("frame %02d: 没有检测到红色方块" % i)

        if args.save_every and i % args.save_every == 0:
            cv2.imwrite(os.path.join(out_dir, "vision_%03d.jpg" % i),
                        annotate(frame, cube, dist))
        time.sleep(args.interval)

    cam.close()
    print("\n[vision] %d/%d 帧命中方块" % (hits, args.frames))
    return 0


def cmd_calib(args):
    """已知距离下测边长，反解距离系数 K。"""
    cam = Camera()
    sides = []
    print("[calib] 请把红色方块摆在相机正前方 %.0f cm 处，保持不动..." % args.dist)
    time.sleep(args.settle)
    for i in range(args.frames):
        frame = cam.read()
        cube = pick_target(find_cubes(frame, args.min_area))
        if cube:
            sides.append(cube["side"])
            print("[calib] %02d: side=%.1fpx" % (i, cube["side"]))
        time.sleep(args.interval)
    cam.close()

    if not sides:
        print("[calib] 一帧都没检测到，先调 HSV/光照，或用 vision 模式排查")
        return 1

    side = float(np.median(sides))
    k = args.dist * side
    print("\n[calib] 边长中位数 %.1fpx @ %.0fcm" % (side, args.dist))
    print("[calib] 距离系数 K = %.0f  （写进脚本的 CUBE_K 即可）" % k)
    return 0


def cmd_grab(args):
    """完整流程：趴下 → 转向对准/靠近 → 抓取 → 起身。

    --dry 时只伸臂到抓取位拍一张照片，不合爪，用来核对夹爪位置。
    """
    dog = dog_connect()
    cam = None
    try:
        dog.translation("z", 95)
        time.sleep(0.8)
        lie_down(dog)

        cam = Camera()
        for step in range(MAX_STEPS):
            frame = cam.read()
            cube = pick_target(find_cubes(frame, args.min_area))

            if cube is None:
                print("[%02d] 没看到方块，退一点再看" % step)
                dog.move_x(-4)
                time.sleep(args.step_pause + 0.15)
                dog.stop()
                continue

            dist = distance_cm(cube["side"], args.k)
            x_err = cube["cx"] - PANEL_CX
            print("[%02d] cx=%d cy=%d side=%.0fpx d=%.1fcm err=%+dpx"
                  % (step, cube["cx"], cube["cy"], cube["side"], dist, x_err))
            cv2.imwrite(os.path.join(args.save_dir, "grab_%02d.jpg" % step),
                        annotate(frame, cube, dist))

            if step_towards(dog, cube, dist, args):
                if args.dry:
                    arm_preview(dog, cam, os.path.join(args.save_dir, "arm_pose.jpg"))
                    print("[dry] 预演完成，未合爪")
                else:
                    grasp(dog, args.claw_close)
                    frame = cam.read()
                    cv2.imwrite(os.path.join(args.save_dir, "after_grab.jpg"),
                                annotate(frame, pick_target(find_cubes(frame))))
                    print("[done] 抓取动作已执行，留档 %s/after_grab.jpg" % args.save_dir)
                break
        else:
            print("[done] 达到最大步数仍未就位，放弃抓取")

        if not args.dry:
            dog.action(2)       # 起身
            time.sleep(2.5)
        return 0
    finally:
        try:
            dog.stop()
        except Exception:
            pass
        if cam:
            cam.close()


# ---------------------------------------------------------------- 入口 --

def build_parser():
    p = argparse.ArgumentParser(
        description="XGO 红色方块视觉识别与抓取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("mode", choices=["vision", "calib", "grab"])
    p.add_argument("--frames", type=int, default=30, help="采样帧数")
    p.add_argument("--interval", type=float, default=0.15, help="帧间隔(秒)")
    p.add_argument("--save-dir", default="/home/pi/red_cube_shots", help="标注图保存目录")
    p.add_argument("--save-every", type=int, default=5, help="每 N 帧存一张(0=不存)")
    p.add_argument("--min-area", type=int, default=MIN_AREA_PX, help="最小轮廓面积")
    p.add_argument("--k", type=float, default=CUBE_K, help="距离系数 K")
    p.add_argument("--near", dest="near", type=float, default=NEAR_CM, help="抓取窗口近端(cm)")
    p.add_argument("--far", dest="far", type=float, default=FAR_CM, help="抓取窗口远端(cm)")
    p.add_argument("--center-tol", dest="center_tol", type=int, default=CENTER_TOL_PX,
                   help="水平居中容差(px)")
    p.add_argument("--step-pause", dest="step_pause", type=float, default=STEP_PAUSE,
                   help="每步停顿(秒)")
    p.add_argument("--claw-close", dest="claw_close", type=int, default=200,
                   help="合爪力度 0-255")
    p.add_argument("--dry", action="store_true",
                   help="抓取预演：只伸臂到抓取位拍照，不合爪")
    p.add_argument("--dist", type=float, default=30.0, help="calib 模式的已知距离(cm)")
    p.add_argument("--settle", type=float, default=2.0, help="calib 模式等待稳定(秒)")
    return p


def main():
    args = build_parser().parse_args()
    if args.mode == "grab":
        os.makedirs(args.save_dir, exist_ok=True)
        return cmd_grab(args)
    if args.mode == "calib":
        return cmd_calib(args)
    return cmd_vision(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[中断] 用户取消")
        sys.exit(130)
    except Exception as exc:
        print("[错误] %s" % exc)
        sys.exit(1)
