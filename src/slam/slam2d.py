#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGO 机器狗 —— 2D 激光 SLAM（纯扫描匹配推定，无里程计）

数据源：狗上 webview.py 的 /frames，或本地 JSONL 录包
算法：
  1. 用阻尼常速模型预测下一帧位姿，IMU 的可信增量可给航向初值
  2. 相关性扫描匹配（Correlative Scan Matching）粗到精搜索最优 (dx, dy, dθ)
     —— 打分用地图的似然场（距离变换），不是和上一帧硬配，所以漂移小
  3. 通过置信度和突跳检查的扫描写入占据栅格（log-odds）
  4. 本机起一个网页看实时地图

用法：
    python3 slam2d.py                       # 默认连 172.18.11.103:8000
    python3 slam2d.py --host 172.18.11.103 --port 8000
    python3 slam2d.py --no-view             # 只建图不出网页
然后浏览器打开 http://127.0.0.1:8010

原理上这套只适合小场景（十几米内）；场景一大、回环一多就需要后端位姿图优化。
"""

import argparse
from bisect import bisect_right
from io import BytesIO
import json
import math
import threading
import time
from collections import deque

import numpy as np
import requests
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, map_coordinates

# ------------------------------------------------------------------ 参数 --

GRID_RES = 0.05          # 栅格分辨率 (m/cell)
GRID_SIZE = 1000         # 栅格边长 (cell) → 50 m × 50 m
USE_RANGE = 6.0          # 参与匹配的最远距离 (m)
MIN_RANGE = 0.23         # 排除狗头和机械臂的近距离回波

SCORE_SIGMA = 0.15       # 似然场高斯宽度 (m)

LOG_ODD_FREE = -0.40
LOG_ODD_OCC = 0.85
LOG_ODD_MIN, LOG_ODD_MAX = -4.0, 4.0

FIELD_REFRESH_EVERY = 5  # 每 N 帧重建一次似然场
IMU_MAX_AGE = 0.15       # 秒，超时姿态不参与匹配

TILT_LIMIT = 25.0        # |roll| 或 |pitch| 超过这个角度就认为狗被抬起/倾倒，该帧丢弃
MOTION_PENALTY = 0.24
MIN_MATCH_SCORE = 0.40
MIN_SCORE_GAIN = 0.004
MIN_TRANSLATION = 0.006
MIN_ROTATION = math.radians(0.25)


# ------------------------------------------------------------ 占据栅格 --

class OccupancyGrid:
    def __init__(self, size=GRID_SIZE, res=GRID_RES):
        self.size = size
        self.res = res
        self.log_odds = np.zeros((size, size), dtype=np.float32)
        self.origin = -size * res / 2.0          # 世界坐标下限
        self.updated = True

    def world_to_cell(self, x, y):
        """返回 (...,2) 的 (col,row) 整型索引。"""
        c0 = np.floor((np.asarray(x) - self.origin) / self.res).astype(np.int32)
        c1 = np.floor((np.asarray(y) - self.origin) / self.res).astype(np.int32)
        return np.stack([c0, c1], axis=-1)

    def cell_to_world(self, c0, c1):
        return (c0 + 0.5) * self.res + self.origin, (c1 + 0.5) * self.res + self.origin

    def integrate(self, pts_world, sensor_xy):
        """把一帧点云写进栅格：端点标占据，射线沿途标空闲。"""
        if len(pts_world) == 0:
            return

        def indices(x, y):
            cells = self.world_to_cell(x, y)
            valid = np.all((cells >= 0) & (cells < self.size), axis=-1)
            cells = cells[valid]
            return np.unique(cells[:, 1] * self.size + cells[:, 0])

        origin = np.asarray(sensor_xy)
        vectors = pts_world - origin
        far = np.linalg.norm(vectors, axis=1).max()
        steps = max(1, math.ceil(far / self.res))
        fractions = np.arange(steps) / steps
        hits = indices(pts_world[:, 0], pts_world[:, 1])
        free = indices(origin[0] + vectors[:, 0, None] * fractions,
                       origin[1] + vectors[:, 1, None] * fractions)
        # 每帧每格只计一次；端点即使也被其他射线经过，仍按命中更新。
        free = np.setdiff1d(free, hits, assume_unique=True)
        cells = self.log_odds.ravel()
        cells[hits] = np.minimum(cells[hits] + LOG_ODD_OCC, LOG_ODD_MAX)
        cells[free] = np.maximum(cells[free] + LOG_ODD_FREE, LOG_ODD_MIN)
        self.updated = True

    def likelihood_field(self):
        """占据格的欧氏距离 → 高斯似然场，用于扫描匹配打分。"""
        occ = self.log_odds > 0.5
        if not occ.any():
            return None
        rows, cols = np.nonzero(occ)
        pad = math.ceil(8 * SCORE_SIGMA / self.res)
        r0, r1 = max(0, rows.min() - pad), min(self.size, rows.max() + pad + 1)
        c0, c1 = max(0, cols.min() - pad), min(self.size, cols.max() + pad + 1)
        dist = distance_transform_edt(~occ[r0:r1, c0:c1]) * self.res
        field = np.zeros_like(self.log_odds)
        field[r0:r1, c0:c1] = np.exp(-0.5 * (dist / SCORE_SIGMA) ** 2)
        return field

    def to_image(self, trajectory):
        """渲染成 RGB 图：空闲白、占据黑、未知灰。

        整张栅格是 50m×50m，直接把全图缩到 1000px 会让几米大的场景只剩几十像素，
        看起来就像"数据是乱的"。所以先按已探测区域裁一刀并留 1.5m 边距，
        再画轨迹，最后整体放大到不超过 1000px。
        """
        lo = self.log_odds
        img = np.full((self.size, self.size, 3), 128, dtype=np.uint8)
        img[lo < -0.4] = (255, 255, 255)     # 空闲
        img[lo > 0.5] = (20, 20, 20)         # 占据
        img[lo > 2.0] = (0, 0, 0)            # 高置信占据

        # 裁到"已知区域 + 边距"，没有已知区域就退回整幅
        known = np.argwhere(np.abs(lo) > 0.4)
        if len(known):
            margin = int(1.5 / self.res)
            r0 = max(0, known[:, 0].min() - margin)
            r1 = min(self.size, known[:, 0].max() + margin + 1)
            c0 = max(0, known[:, 1].min() - margin)
            c1 = min(self.size, known[:, 1].max() + margin + 1)
        else:
            r0, r1, c0, c1 = 0, self.size, 0, self.size

        im = Image.fromarray(img[r0:r1, c0:c1]).transpose(Image.FLIP_TOP_BOTTOM)
        dr = ImageDraw.Draw(im)
        h = r1 - r0

        def to_px(x, y):
            cc, rr = self.world_to_cell(x, y)
            return int(cc - c0), int(h - 1 - (rr - r0))

        if len(trajectory) > 1:
            dr.line([to_px(x, y) for x, y, _ in trajectory], fill=(30, 120, 255), width=2)
        if trajectory:
            x, y, th = trajectory[-1]
            cx, cy = to_px(x, y)
            r = max(6, min(im.size) // 60)
            dr.ellipse([cx - r / 2, cy - r / 2, cx + r / 2, cy + r / 2], fill=(230, 60, 60))
            dr.line([cx, cy, cx + 2 * r * math.cos(th), cy - 2 * r * math.sin(th)],
                    fill=(230, 60, 60), width=2)

        # 放大到便于观看的尺寸（最近邻，保持栅格边界清晰）
        scale = max(1, min(1000 // max(im.size), 8))
        if scale > 1:
            im = im.resize((im.size[0] * scale, im.size[1] * scale), Image.NEAREST)
        return im


# ------------------------------------------------------------ 扫描匹配 --

class ScanMatcher:
    """粗到精的相关性扫描匹配。"""

    def __init__(self, grid):
        self.grid = grid
        self.field = None
        self._frame = 0
        coarse_xy = np.arange(-4, 5) * 0.075
        fine_xy = np.arange(-2, 3) * 0.021
        self.coarse = self._offsets(coarse_xy, np.arange(-7, 8) * 0.048)
        self.fine = self._offsets(fine_xy, np.arange(-2, 3) * 0.016)
        self.coarse_xy = self.coarse[self.coarse[:, 2] == 0]
        self.fine_xy = self.fine[self.fine[:, 2] == 0]

    @staticmethod
    def _offsets(xy, theta):
        return np.stack(np.meshgrid(xy, xy, theta, indexing="ij"), axis=-1).reshape(-1, 3)

    def maybe_refresh(self):
        if self.field is None or self._frame % FIELD_REFRESH_EVERY == 0:
            f = self.grid.likelihood_field()
            if f is not None:
                self.field = f
        self._frame += 1

    def _score(self, pts, hyp, field):
        """hyp: (H,3) 位姿候选；pts: (N,2) 雷达坐标系点。返回 (H,) 得分。"""
        x, y, th = hyp[:, 0], hyp[:, 1], hyp[:, 2]
        c, s = np.cos(th), np.sin(th)

        px, py = pts[:, 0], pts[:, 1]
        # (H,N) 世界坐标
        wx = c[:, None] * px[None, :] - s[:, None] * py[None, :] + x[:, None]
        wy = s[:, None] * px[None, :] + c[:, None] * py[None, :] + y[:, None]

        # 栅格值在格心，亚栅格插值让精搜索不受取整台阶影响。
        cols = (wx - self.grid.origin) / self.grid.res - 0.5
        rows = (wy - self.grid.origin) / self.grid.res - 0.5
        return map_coordinates(field, (rows, cols), order=1, prefilter=False).mean(axis=1)

    def search(self, pts, guess, search_theta=True):
        """在 guess 附近做粗到精搜索，返回最优位姿。"""
        if self.field is None or len(pts) == 0:
            return np.array(guess, dtype=np.float64)

        sample = pts[::3]
        coarse = self.coarse if search_theta else self.coarse_xy
        fine = self.fine if search_theta else self.fine_xy
        best = np.asarray(guess)
        for offsets in (coarse, fine, 0.35 * fine):
            candidates = best + offsets
            penalty = MOTION_PENALTY * np.sum((candidates - guess) ** 2, axis=1)
            scores = self._score(sample, candidates, self.field) - penalty
            best = candidates[np.argmax(scores)]
        return best


# ---------------------------------------------------------------- 数据源 --

def polar_to_xy(points, max_range=USE_RANGE):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ranges = points[:, 1]
    keep = np.isfinite(points).all(axis=1) & (ranges > MIN_RANGE) & (ranges < max_range)
    angles = np.deg2rad(points[keep, 0])
    return ranges[keep, None] * np.column_stack((np.cos(angles), np.sin(angles)))


class ScanSource:
    """按顺序消费狗端 /frames 的批量帧（每帧自带采集时间戳）。

    旧接口 /scan 只给"最新一帧快照"：网络一抖，客户端要么拿到重复帧、要么直接
    跳过若干帧，而且不知道这帧是什么时候采的。改成批量拉取 + 队列后：
      - 帧按 id 严格递增消费，不重复、不回退；
      - 每帧带狗端时间戳，客户端可以用真实 Δt 做运动预测；
      - 积压过多时丢最旧的帧，保证实时性（会记 dropped 计数）。
    """

    def __init__(self, host, port, max_range=USE_RANGE):
        self.url = "http://%s:%d/frames" % (host, port)
        self.max_range = max_range
        self.last_id = -1
        self.lock = threading.Lock()
        self.queue = deque(maxlen=120)
        self.alive = True
        self.hz = 0.0
        self.dropped = 0
        self.last_pts = []          # 最新一帧原始点 [[deg, m], ...]，给实时雷达图用
        self.last_imu = None        # 最新一帧对应的 IMU [roll, pitch, yaw, t]
        self.last_id = self._seed_last_id(host, port)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _seed_last_id(host, port):
        """先问一次 /scan 拿到当前帧号，避免启动时把狗端缓冲的几十帧历史全灌进来。"""
        try:
            d = requests.get("http://%s:%d/scan" % (host, port), timeout=2.0).json()
            return int(d["scan_id"])
        except Exception:
            return -1

    def _loop(self):
        cnt, t0 = 0, time.time()
        while self.alive:
            try:
                d = requests.get(self.url, params={"since": self.last_id, "max": 30},
                                 timeout=2.0).json()
            except Exception:
                time.sleep(0.15)
                continue
            for f in d.get("frames", []):
                fid = int(f["id"])
                if fid <= self.last_id:
                    continue
                self.last_id = fid
                xy = polar_to_xy(f["pts"], self.max_range)
                imu = f.get("imu")
                stamp = float(f["t"])
                fresh = bool(imu) and 0 <= stamp - imu[3] <= IMU_MAX_AGE
                yaw = float(imu[2]) if fresh else None
                skip = fresh and (abs(imu[0]) > TILT_LIMIT or abs(imu[1]) > TILT_LIMIT)
                with self.lock:
                    self.queue.append((xy, fid, stamp, yaw, skip))
                    self.last_pts = f["pts"]
                    self.last_imu = imu
                cnt += 1
            now = time.time()
            if now - t0 >= 1.0:
                self.hz = cnt / (now - t0)
                cnt, t0 = 0, now
            time.sleep(0.05)

    def get(self):
        with self.lock:
            if not self.queue:
                return None
            if len(self.queue) > 15:      # 积压太多，丢旧帧保实时（只留最新 5 帧）
                self.dropped += len(self.queue) - 5
                for _ in range(len(self.queue) - 5):
                    self.queue.popleft()
            return self.queue.popleft()


class ReplaySource:
    """离线回放：支持两种格式，都当数据源用。

    1) 录包格式（record_bag.py 产出）：{"type":"scan","t":..,"id":..,"pts":[[deg,m],..]}
       和 {"type":"imu","t":..,"r":..,"p":..,"y":..}；回放时按时间戳取最近的 IMU yaw。
    2) 旧格式（slam2d --record 产出）：{"id":..,"t":..,"yaw":..,"xy":[[x,y],..]}
    """

    def __init__(self, path):
        self.frames = []
        self.imu = []
        with open(path, encoding="utf-8") as bag:
            for line in bag:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("type") == "scan" or "xy" in r:
                    self.frames.append(r)
                elif r.get("type") == "imu":
                    self.imu.append((float(r["t"]), float(r.get("r", 0)),
                                     float(r.get("p", 0)), float(r["y"])))
        self.imu.sort()
        self.i = 0
        self.alive = True
        self.dropped = 0
        self.lock = threading.Lock()
        self.last_pts = []
        self.last_imu = None
        dur = (self.frames[-1]["t"] - self.frames[0]["t"]) if self.frames else 0
        self.hz = (len(self.frames) - 1) / dur if dur > 0 else 0.0
        print("[replay] %s：%d 帧 / %.1f s（%.1f Hz），IMU %d 样本"
              % (path, len(self.frames), dur, self.hz, len(self.imu)))

    def _imu_at(self, t):
        """取 t 之前（含）最近的 IMU 样本 (t, roll, pitch, yaw)；太旧则视为无效。"""
        if not self.imu:
            return None
        i = bisect_right(self.imu, (t, float("inf"))) - 1
        if i < 0 or t - self.imu[i][0] > IMU_MAX_AGE:
            return None
        return self.imu[i]

    def get(self):
        if self.i >= len(self.frames):
            self.alive = False
            return None
        f = self.frames[self.i]
        self.i += 1
        if "pts" in f:
            points = f["pts"]
            xy = polar_to_xy(points)
        else:
            xy = np.asarray(f["xy"], dtype=np.float64).reshape(-1, 2)
            deg = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
            rng = np.hypot(xy[:, 0], xy[:, 1])
            points = np.round(np.column_stack((deg, rng)), 3).tolist()
            xy = xy[(rng > MIN_RANGE) & (rng < USE_RANGE)]
        t = float(f["t"])
        smp = self._imu_at(t)
        yaw = f.get("yaw")
        if yaw is None and smp is not None:
            yaw = smp[3]
        skip = bool(smp) and (abs(smp[1]) > TILT_LIMIT or abs(smp[2]) > TILT_LIMIT)
        with self.lock:
            self.last_pts = points
            self.last_imu = [*smp[1:], smp[0]] if smp else None
        return xy, int(f["id"]), t, yaw, skip


# ---------------------------------------------------------------- 主流程 --

class SLAM:
    def __init__(self, source, use_imu=True, imu_heading=False, imu_sign=1.0):
        self.src = source
        self.use_imu = use_imu
        self.imu_heading = imu_heading
        # 雷达角已统一成逆时针后，IMU yaw 与之一致，符号取 +1
        self.imu_sign = imu_sign
        self.yaw0 = None
        self.th0 = 0.0
        self.grid = OccupancyGrid()
        self.matcher = ScanMatcher(self.grid)
        self.pose = np.array([0.0, 0.0, 0.0])
        self.prev_pose = np.array([0.0, 0.0, 0.0])
        self.traj = deque(maxlen=5000)
        self.frames = 0
        self.last_imu_yaw = None
        self.imu_used = 0
        self.last_id = -1
        self.last_t = None
        self.prev_dt = 0.1
        self.rec = None             # 记录文件对象（--record）
        self.skipped = 0            # 因姿态倾斜过大而跳过
        self.rejected = 0           # 匹配置信度不足或位姿跳变
        self.map_img = None
        self.trace = None           # 置为 list 时逐帧记录决策（离线诊断用）
        self.lock = threading.Lock()

    def step(self):
        f = self.src.get()
        if f is None:
            return False
        pts, sid, t, yaw, *flags = f
        if sid <= self.last_id:
            return False
        self.last_id = sid
        if flags and flags[0]:
            self.skipped += 1
            self.prev_pose = self.pose
            self.last_imu_yaw = None
            return False

        if len(pts) < 10:
            return False

        # 用狗端时间戳算真实帧间隔（网络抖动不会伪装成"机器人移动变快/变慢"）
        dt = self.prev_dt if self.last_t is None else float(np.clip(t - self.last_t, 0.02, 0.5))
        self.last_t = t
        if self.rec is not None:
            self.rec.write(json.dumps({"id": sid, "t": round(t, 3),
                                       "yaw": yaw,
                                       "xy": np.round(pts, 3).tolist()}) + "\n")

        # 常量速度预测（按 Δt 缩放）
        scale = min(1.25, dt / max(self.prev_dt, 0.02)) * 0.8
        self.prev_dt = dt
        guess = self.pose + (self.pose - self.prev_pose) * scale
        # IMU 航向先验：转角明显时用 IMU 的 Δyaw 覆盖预测的旋转分量
        dth_imu = None
        if self.use_imu and yaw is not None and self.last_imu_yaw is not None:
            dth_imu = self.imu_sign * ((yaw - self.last_imu_yaw + 180.0) % 360.0 - 180.0)
            if 0.5 < abs(dth_imu) <= 30.0:
                guess[2] = self.pose[2] + math.radians(dth_imu)
                self.imu_used += 1
        self.last_imu_yaw = yaw
        # 模式 B：航向直接由 IMU 给（θ = θ0 + Δyaw），扫描匹配只解平移
        search_theta = True
        if self.use_imu and self.imu_heading and yaw is not None:
            if self.yaw0 is None:
                self.yaw0, self.th0 = yaw, self.pose[2]
            heading = self.th0 + self.imu_sign * math.radians(yaw - self.yaw0)
            guess[2] = self.pose[2] + (heading - self.pose[2] + np.pi) % (2 * np.pi) - np.pi
            search_theta = False

        self.matcher.maybe_refresh()
        new_pose = self.matcher.search(pts, guess, search_theta=search_theta)
        trace = {"t": t, "yaw": yaw, "dth_imu": dth_imu,
                 "guess_th": float(np.degrees(guess[2])),
                 "match_th": float(np.degrees(new_pose[2])),
                 "gate": False, "jump": False, "dead": False,
                 "s_new": None, "s_old": None}

        if self.matcher.field is not None:
            s_new, s_old = self.matcher._score(pts, np.stack((new_pose, self.pose)), self.matcher.field)
            s_new, s_old = float(s_new), float(s_old)
            trace["s_new"], trace["s_old"] = s_new, s_old
            if s_new < s_old + MIN_SCORE_GAIN:
                new_pose = self.pose.copy()
                trace["gate"] = True

        # 置信度不足或突跳时保留上一可信位姿和地图。
        d = np.hypot(new_pose[0] - self.pose[0], new_pose[1] - self.pose[1])
        dth = abs((new_pose[2] - self.pose[2] + np.pi) % (2 * np.pi) - np.pi)
        score = trace["s_new"]
        weak_match = score is not None and max(score, trace["s_old"]) < MIN_MATCH_SCORE
        if d > 0.6 or dth > 0.75 or weak_match:
            self.rejected += 1
            self.prev_pose = self.pose
            if self.trace is not None:
                trace["jump"] = True
                self.trace.append(trace)
            return False

        moving = d >= MIN_TRANSLATION or dth >= MIN_ROTATION
        if not moving:
            new_pose = self.pose.copy()
            trace["dead"] = True

        self.prev_pose = self.pose
        self.pose = new_pose

        map_changed = moving or self.frames % FIELD_REFRESH_EVERY == 0
        if map_changed:
            c, s = math.cos(new_pose[2]), math.sin(new_pose[2])
            world = pts @ np.array([[c, s], [-s, c]]) + new_pose[:2]
            self.grid.integrate(world, new_pose[:2])
        self.traj.append(tuple(float(v) for v in new_pose))
        self.frames += 1
        if self.trace is not None:
            trace["x"], trace["y"] = float(new_pose[0]), float(new_pose[1])
            trace["th"] = float(np.degrees(new_pose[2]))
            self.trace.append(trace)

        if map_changed:
            rendered = self.grid.to_image(list(self.traj))
            with self.lock:
                self.map_img = rendered
        return True

    def map_png(self, blank_size=100):
        with self.lock:
            image = self.map_img
        if image is None:
            image = Image.new("RGB", (blank_size, blank_size), (128, 128, 128))
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return buffer.getvalue()

    def stats(self):
        x, y, theta = self.pose
        known_cells = int((np.abs(self.grid.log_odds) > 0.4).sum())
        return {
            "x": float(x), "y": float(y), "th": float(theta),
            "frames": self.frames, "known_m2": known_cells * self.grid.res ** 2,
        }


# ------------------------------------------------------------------ 网页 --

PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>XGO SLAM · 实时雷达 / IMU</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#0d1117;color:#c9d1d9;font:13px/1.5 -apple-system,"PingFang SC",sans-serif}
 header{display:flex;gap:18px;align-items:center;padding:10px 16px;border-bottom:1px solid #21262d}
 h1{font-size:14px;margin:0;font-weight:600}
 .s{font-variant-numeric:tabular-nums;color:#8b949e}
 b{color:#58a6ff;font-weight:600}
 main{display:flex;gap:12px;height:calc(100vh - 43px);padding:12px;box-sizing:border-box}
 .mapbox{flex:1;display:flex;align-items:center;justify-content:center;min-width:0}
 .mapbox img{max-width:100%;max-height:100%;image-rendering:pixelated;border:1px solid #21262d;border-radius:4px}
 aside{width:400px;flex:0 0 400px;display:flex;flex-direction:column;gap:12px}
 .card{background:#11161d;border:1px solid #21262d;border-radius:6px;padding:8px}
 .card h2{font-size:12px;margin:0 0 6px;color:#8b949e;font-weight:500;letter-spacing:.5px}
 #radar,#horizon{display:block;margin:0 auto;background:#0d1117;border-radius:4px}
 .imu{display:flex;gap:8px;align-items:center}
 .nums{flex:1;display:grid;grid-template-columns:auto 1fr;gap:2px 8px;font-variant-numeric:tabular-nums}
 .nums span{color:#8b949e}
 .nums b{color:#e6edf3;font-weight:600;text-align:right}
 .yaw{color:#58a6ff!important}
</style></head><body>
<header>
 <h1>XGO · SLAM</h1>
 <span class="s">位姿 <b id="pose">—</b></span>
 <span class="s">帧 <b id="frames">0</b></span>
 <span class="s">雷达 <b id="hz">—</b></span>
 <span class="s">面积 <b id="area">—</b></span>
 <span class="s">丢帧 <b id="drop">0</b></span>
</header>
<main>
 <div class="mapbox"><img id="map" src="/map.png"></div>
 <aside>
  <div class="card"><h2>实时雷达（0° = 车头方向，量程自适应）</h2>
    <canvas id="radar" width="380" height="380"></canvas>
  </div>
  <div class="card"><h2>IMU 姿态</h2>
   <div class="imu">
    <canvas id="horizon" width="150" height="150"></canvas>
    <div class="nums">
      <span>roll</span><b id="roll">—</b>
      <span>pitch</span><b id="pitch">—</b>
      <span>yaw</span><b class="yaw" id="yaw">—</b>
      <span>yaw 角速度</span><b id="yawrate">—</b>
    </div>
   </div>
  </div>
 </aside>
</main>
<script>
const radar=document.getElementById('radar'), rctx=radar.getContext('2d');
const hor=document.getElementById('horizon'), hctx=hor.getContext('2d');
const MAXR=6.0;
let lastYaw=null, lastYawT=null, yawRate=0;

function drawRadar(pts){
  const W=radar.width,H=radar.height,cx=W/2,cy=H/2;
  // 量程自适应：按当前帧最远点取整，最小 2 m，最大 16 m
  let maxR=2;
  if(pts&&pts.length){ let mx=0; for(let i=0;i<pts.length;i++){ if(pts[i][1]>mx) mx=pts[i][1]; }
    maxR=Math.min(16,Math.max(2,Math.ceil(mx))); }
  window._maxR=maxR;
  const scale=(Math.min(W,H)/2-16)/maxR;
  rctx.fillStyle='#0d1117'; rctx.fillRect(0,0,W,H);
  rctx.strokeStyle='#1b2431'; rctx.lineWidth=1;
  rctx.fillStyle='#3d4b5c'; rctx.font='10px sans-serif';
  const step=maxR/4;
  for(let k=1;k<=4;k++){
    const r=step*k;
    rctx.beginPath(); rctx.arc(cx,cy,r*scale,0,Math.PI*2); rctx.stroke();
    rctx.fillText(r.toFixed(r<10?1:0)+'m',cx+2,cy-r*scale-2);
  }
  rctx.strokeStyle='#243040';
  rctx.beginPath(); rctx.moveTo(cx,cy-maxR*scale); rctx.lineTo(cx,cy+maxR*scale);
  rctx.moveTo(cx-maxR*scale,cy); rctx.lineTo(cx+maxR*scale,cy); rctx.stroke();
  if(!pts||!pts.length){ rctx.fillStyle='#484f58'; rctx.fillText('等待雷达数据…',12,20); return; }
  for(let i=0;i<pts.length;i++){
    const deg=pts[i][0], m=pts[i][1];
    if(m<=0.02) continue;
    const a=deg*Math.PI/180;
    const x=cx-m*scale*Math.sin(a), y=cy-m*scale*Math.cos(a);
    const t=Math.min(1,m/(window._maxR||6));
    rctx.fillStyle='rgb('+Math.round(90+150*t)+','+Math.round(220-110*t)+',255)';
    rctx.fillRect(x-1,y-1,2,2);
  }
  // 车头三角
  rctx.fillStyle='#f85149';
  rctx.beginPath(); rctx.moveTo(cx,cy-7); rctx.lineTo(cx-5,cy+5); rctx.lineTo(cx+5,cy+5);
  rctx.closePath(); rctx.fill();
}

function drawHorizon(roll,pitch){
  const W=hor.width,H=hor.height,cx=W/2,cy=H/2,R=W/2-4;
  hctx.clearRect(0,0,W,H);
  hctx.save();
  hctx.beginPath(); hctx.arc(cx,cy,R,0,Math.PI*2); hctx.clip();
  hctx.translate(cx,cy); hctx.rotate(roll*Math.PI/180);
  hctx.translate(0, pitch*1.8);
  hctx.fillStyle='#12395e'; hctx.fillRect(-W,-H,W,2*H);
  hctx.fillStyle='#3a2a1c'; hctx.fillRect(-W,0,W,2*H);
  hctx.strokeStyle='#e6edf3'; hctx.lineWidth=1.5;
  hctx.beginPath(); hctx.moveTo(-W,0); hctx.lineTo(W,0); hctx.stroke();
  hctx.restore();
  hctx.strokeStyle='#30363d'; hctx.lineWidth=2;
  hctx.beginPath(); hctx.arc(cx,cy,R,0,Math.PI*2); hctx.stroke();
  hctx.strokeStyle='#f0f6fc'; hctx.lineWidth=2;
  hctx.beginPath(); hctx.moveTo(cx-18,cy); hctx.lineTo(cx-6,cy);
  hctx.moveTo(cx+6,cy); hctx.lineTo(cx+18,cy); hctx.moveTo(cx,cy); hctx.lineTo(cx,cy); hctx.stroke();
  hctx.fillStyle='#f0f6fc'; hctx.fillRect(cx-1.5,cy-1.5,3,3);
}

async function live(){
  try{
    const d=await (await fetch('/live.json',{cache:'no-store'})).json();
    drawRadar(d.scan);
    const imu=d.imu;
    if(imu){
      const [r,p,y,t]=imu;
      document.getElementById('roll').textContent=r.toFixed(2)+'°';
      document.getElementById('pitch').textContent=p.toFixed(2)+'°';
      document.getElementById('yaw').textContent=y.toFixed(2)+'°';
      if(lastYaw!==null && t>lastYawT){
        let dy=((y-lastYaw+540)%360)-180;
        yawRate=dy/(t-lastYawT);
      }
      lastYaw=y; lastYawT=t;
      document.getElementById('yawrate').textContent=yawRate.toFixed(1)+'°/s';
      drawHorizon(r,p);
    }
    document.getElementById('hz').textContent=(d.hz||0).toFixed(1)+' Hz';
    document.getElementById('drop').textContent=d.dropped||0;
  }catch(e){}
  setTimeout(live,100);
}

async function stats(){
  try{
    const d=await (await fetch('/stats',{cache:'no-store'})).json();
    document.getElementById('pose').textContent=
      'x='+d.x.toFixed(2)+' y='+d.y.toFixed(2)+' θ='+(d.th*180/Math.PI).toFixed(1)+'°';
    document.getElementById('frames').textContent=d.frames;
    document.getElementById('area').textContent=d.known_m2.toFixed(1)+' m²';
    if(d.frames!==window._last){
      window._last=d.frames;
      document.getElementById('map').src='/map.png?t='+Date.now();
    }
  }catch(e){}
  setTimeout(stats,800);
}
live(); stats();
</script></body></html>"""


def run_server(slam, source, port, host="127.0.0.1"):
    from flask import Flask, Response, jsonify

    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE

    @app.route("/map.png")
    def map_png():
        return Response(slam.map_png(), mimetype="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.route("/live.json")
    def live_json():
        with source.lock:
            pts = list(source.last_pts or [])
            imu = source.last_imu
        return jsonify({
            "scan": pts, "imu": imu,
            "x": float(slam.pose[0]), "y": float(slam.pose[1]), "th": float(slam.pose[2]),
            "frames": slam.frames, "hz": float(source.hz),
            "dropped": int(getattr(source, "dropped", 0)),
        })

    @app.route("/stats")
    def stats():
        return jsonify({
            **slam.stats(),
            "hz": float(source.hz),
            "imu_on": bool(slam.use_imu), "imu_used": slam.imu_used,
            "skipped": slam.skipped, "rejected": slam.rejected,
        })

    print("[slam] 网页 http://%s:%d" % (host, port))
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.18.11.103")
    ap.add_argument("--port", type=int, default=8000, help="狗端 webview 端口")
    ap.add_argument("--view-port", type=int, default=8010, help="本机网页端口")
    ap.add_argument("--bind", default="127.0.0.1", help="网页监听地址（狗上跑时用 0.0.0.0）")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--no-imu", action="store_true", help="不用 IMU 航向先验（A/B 对比用）")
    ap.add_argument("--imu-sign", type=float, default=1.0,
                    help="IMU yaw 相对雷达坐标系的符号；雷达角已统一为逆时针后取 +1")
    ap.add_argument("--imu-heading", action="store_true",
                    help="诊断模式：航向完全交给 IMU，扫描匹配只解平移")
    ap.add_argument("--record", metavar="FILE", help="把消费到的帧存成 JSONL，供离线回放")
    ap.add_argument("--replay", metavar="FILE", help="离线回放 JSONL（不再连狗）")
    ap.add_argument("--map-out", metavar="FILE", default="/tmp/slam_map.png", help="退出时保存的地图 PNG")
    args = ap.parse_args()

    if args.replay:
        src = ReplaySource(args.replay)
    else:
        src = ScanSource(args.host, args.port)
        print("[slam] 数据源 %s:%d" % (args.host, args.port))
        time.sleep(1.5)
    slam = SLAM(src, use_imu=not args.no_imu, imu_heading=args.imu_heading,
                imu_sign=args.imu_sign)
    print("[slam] IMU 航向先验：%s" % ("开" if slam.use_imu else "关"))
    if args.record:
        slam.rec = open(args.record, "w")
        print("[slam] 记录到 %s" % args.record)

    if not args.no_view:
        threading.Thread(target=run_server, args=(slam, src, args.view_port, args.bind),
                         daemon=True).start()

    print("[slam] 开始建图，Ctrl+C 退出")
    try:
        while True:
            if slam.step():
                if slam.frames % 20 == 1:
                    print("[slam] 帧 %4d  位姿 (%+.2f, %+.2f, %+.1f°)"
                          % (slam.frames, slam.pose[0], slam.pose[1],
                             math.degrees(slam.pose[2])))
            else:
                if not getattr(src, "alive", True):
                    break
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[slam] 停止")
    finally:
        src.alive = False
        if slam.rec is not None:
            slam.rec.close()
        print("[slam] 处理 %d 帧，倾斜跳过 %d 帧，匹配拒绝 %d 帧，接收丢失 %d 帧"
              % (slam.frames, slam.skipped, slam.rejected, src.dropped))
        with slam.lock:
            if slam.map_img is not None:
                slam.grid.to_image(list(slam.traj)).save(args.map_out)
                print("[slam] 地图已存 %s" % args.map_out)


if __name__ == "__main__":
    main()
