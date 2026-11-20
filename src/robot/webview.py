#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGO 机器狗 —— 摄像头 + 激光雷达 实时网页监视

浏览器打开 http://<狗IP>:8000 即可看到：
  - 左：相机实时画面（MJPEG 流）
  - 右：YDLidar Tmini Plus 的 360° 扫描，实时绘制

跑之前必须先停掉 luwu_launcher，否则摄像头被它的预览进程独占：
    sudo systemctl stop luwu-launcher
    setsid -f python3 -u webview.py > /tmp/webview.log 2>&1 < /dev/null
跑完恢复：
    sudo systemctl start luwu-launcher

实测性能（CM5 / 640x480 / q80）：
    相机 31.8 fps，雷达 10 Hz（约 350 点/帧），Mac→狗 WiFi 约 3.4 MB/s
踩过的坑：
    1. ydlidar 的 doProcessSimple() 在 C 层忙等并持有 GIL。和相机放在同一
       进程的两个线程里，相机帧率会从 32 fps 塌到 0.3 fps。必须放独立进程。
    2. Picamera2 不能被多个消费者同时 capture_array()，要单线程采集。
    3. setsid / nohup 启动后 ssh 会挂着不退，用 setsid -f 才是真 fork。

依赖：Flask、OpenCV、picamera2、ydlidar SDK（都已预装）
"""

import math
import os
import signal
from collections import deque
from queue import Full
import threading
import time

import cv2
from flask import Flask, Response, render_template_string, request, jsonify

from lidar import create_lidar

CAM_W, CAM_H = 640, 480
CAM_FPS = 20
HTTP_PORT = 8000

app = Flask(__name__)


# ------------------------------------------------------------------ 摄像头 --

class Camera:
    """后台线程持续采集并编码，客户端只取最新一帧。

    Picamera2 不宜被多个消费者同时 capture_array()，否则缓冲区会被抢空、
    帧率塌到 1 fps 以下。这里统一由一个线程独占采集。
    """

    def __init__(self, width=CAM_W, height=CAM_H, quality=80):
        from picamera2 import Picamera2
        self.quality = quality
        self.picam = Picamera2()
        cfg = self.picam.create_preview_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        self.picam.configure(cfg)
        self.picam.start()
        time.sleep(0.6)
        for _ in range(5):          # 丢掉前几帧，等 AE/AWB 收敛
            self.picam.capture_array()

        self._jpeg = None
        self._seq = 0
        self._fps = 0.0
        self._lock = threading.Lock()
        threading.Thread(target=self._grab_loop, daemon=True).start()

    def _grab_loop(self):
        cnt, t0 = 0, time.time()
        while True:
            frame = self.picam.capture_array()
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            )
            if not ok:
                continue
            with self._lock:
                self._jpeg = buf.tobytes()
                self._seq += 1
            cnt += 1
            now = time.time()
            if now - t0 >= 1.0:
                self._fps = cnt / (now - t0)
                cnt, t0 = 0, now

    def latest(self):
        with self._lock:
            return self._jpeg, self._seq, self._fps

    def close(self):
        try:
            self.picam.stop()
            self.picam.close()
        except Exception:
            pass


# -------------------------------------------------------------------- 雷达 --

MAX_POINTS = 4096          # 单帧最多点数（Tmini Plus 约 410 点/帧）


def _lidar_worker(arr, lock, q, stop):
    """子进程里跑 ydlidar。

    ydlidar 的 doProcessSimple() 是 C 层忙等轮询，**会一直持有 GIL**，
    实测把它和相机放在同一进程的两个线程里，相机帧率会从 32 fps 掉到 0.3 fps。
    所以必须放进独立进程，靠共享内存把最新一帧传回去。
    """
    ydlidar, laser = create_lidar()

    if not laser.initialize() or not laser.turnOn():
        print("[lidar] 启动失败: " + laser.DescribeError(), flush=True)
        return

    print("[lidar] 子进程开始扫描", flush=True)

    # Tmini Plus 的每次成功调用已返回一圈。按主机的 100 ms 窗口再分箱，
    # 会把时间略短于阈值的那一圈与下一圈混在一起。
    seq = 0

    try:
        while not stop.is_set() and ydlidar.os_isOk():
            scan = ydlidar.LaserScan()
            if not laser.doProcessSimple(scan):
                time.sleep(0.005)
                continue
            now = time.time()
            points = []
            for p in scan.points:
                if math.isfinite(p.range) and p.range > 0.03 and math.isfinite(p.angle):
                    # 雷达原始角顺时针为正，接口统一发布左为正的逆时针角。
                    points.append([round(-math.degrees(p.angle), 2), round(p.range, 3)])
            points = points[:MAX_POINTS]
            seq += 1
            with lock:
                arr[0:3] = [len(points), seq, now]
                arr[3:3 + 2 * len(points)] = [v for point in points for v in point]
            try:
                q.put_nowait({"id": seq, "t": round(now, 3), "pts": points})
            except Full:
                pass  # 队列满时让帧号体现丢失，采集继续运行。
    finally:
        laser.turnOff()
        laser.disconnect()


class LidarProc:
    """雷达子进程的父侧句柄。"""

    def __init__(self):
        import multiprocessing as mp
        self.arr = mp.Array("d", 3 + 2 * MAX_POINTS, lock=False)
        self.lock = mp.Lock()
        self._q = mp.Queue(maxsize=512)
        self._stop = mp.Event()
        self.frames = deque(maxlen=200)      # 最近 ~20s 的帧，供 /frames 批量取
        self.frames_lock = threading.Lock()
        self.proc = mp.Process(target=_lidar_worker,
                               args=(self.arr, self.lock, self._q, self._stop), daemon=True)
        self.proc.start()
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        """把子进程发来的带时间戳的帧收进本地环形缓冲。"""
        while True:
            try:
                f = self._q.get(timeout=0.5)
            except Exception:
                if not self.proc.is_alive():
                    return
                continue
            with self.frames_lock:
                self.frames.append(f)

    def wait_ready(self, timeout=8.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.arr[0] > 0:
                    return True
            time.sleep(0.1)
        return False

    def latest(self):
        """最新一帧 (pts, id, t)；缓冲为空时退回共享内存。"""
        with self.frames_lock:
            if self.frames:
                f = self.frames[-1]
                return [(a, r) for a, r in f["pts"]], f["id"], f["t"]
        with self.lock:
            n = int(self.arr[0])
            seq = int(self.arr[1])
            t = float(self.arr[2])
            n = max(0, min(n, MAX_POINTS))
            pts = [(self.arr[3 + 2 * i], self.arr[4 + 2 * i]) for i in range(n)]
        return pts, seq, t

    def alive(self):
        return self.proc.is_alive()

    def close(self):
        self._stop.set()
        self.proc.join(timeout=3)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join()


# -------------------------------------------------------------------- 页面 --

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>XGO · 相机 + 雷达</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0d1117; color:#c9d1d9;
         font:13px/1.5 -apple-system,"PingFang SC",Helvetica,Arial,sans-serif; }
  header { display:flex; align-items:center; gap:16px; padding:10px 16px;
           border-bottom:1px solid #21262d; }
  h1 { font-size:14px; font-weight:600; margin:0; letter-spacing:.5px; }
  .stat { font-variant-numeric:tabular-nums; color:#8b949e; }
  .dot { width:8px; height:8px; border-radius:50%; background:#3fb950;
         display:inline-block; margin-right:6px; }
  main { display:grid; grid-template-columns:1fr 1fr; gap:1px;
         background:#21262d; height:calc(100vh - 42px); }
  .pane { background:#0d1117; display:flex; flex-direction:column; min-width:0; }
  .pane h2 { margin:0; padding:8px 14px; font-size:12px; font-weight:500;
             color:#8b949e; border-bottom:1px solid #21262d; }
  .body { flex:1; display:flex; align-items:center; justify-content:center;
          overflow:hidden; padding:10px; }
  img, canvas { max-width:100%; max-height:100%; display:block; }
  img { border-radius:4px; }
  .ctl { padding:6px 14px; border-top:1px solid #21262d; display:flex;
         gap:14px; align-items:center; color:#8b949e; font-size:12px; }
  select, input[type=range] { background:#161b22; color:#c9d1d9;
         border:1px solid #30363d; border-radius:4px; padding:2px 6px; font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>XGO · 实时监视</h1>
  <span class="stat"><span class="dot" id="camDot"></span>相机 <span id="camFps">—</span></span>
  <span class="stat"><span class="dot" id="ldDot"></span>雷达 <span id="ldFps">—</span></span>
  <span class="stat">点数 <span id="ldPts">—</span></span>
</header>
<main>
  <section class="pane">
    <h2>CAMERA · ov5647</h2>
    <div class="body"><img id="cam" src="/stream.mjpg" alt="camera"></div>
  </section>
  <section class="pane">
    <h2>LIDAR · YDLidar Tmini Plus 360°</h2>
    <div class="body"><canvas id="radar" width="620" height="620"></canvas></div>
    <div class="ctl">
      <label>量程 <select id="range">
        <option value="2">2 m</option>
        <option value="4" selected>4 m</option>
        <option value="8">8 m</option>
        <option value="12">12 m</option>
      </select></label>
      <label>点大小 <input id="psize" type="range" min="1" max="4" value="2"></label>
    </div>
  </section>
</main>
<script>
const cv = document.getElementById('radar'), ctx = cv.getContext('2d');
const camImg = document.getElementById('cam');
let rangeM = 4, psize = 2;

document.getElementById('range').onchange = e => rangeM = +e.target.value;
document.getElementById('psize').oninput = e => psize = +e.target.value;

// ---- 相机帧率统计 ----
let camFrames = 0, camT0 = performance.now();
camImg.onload = () => {
  camFrames++;
  const now = performance.now();
  if (now - camT0 > 1000) {
    document.getElementById('camFps').textContent =
      (camFrames * 1000 / (now - camT0)).toFixed(1) + ' fps';
    camFrames = 0; camT0 = now;
    document.getElementById('camDot').style.background = '#3fb950';
  }
};
camImg.onerror = () => document.getElementById('camDot').style.background = '#f85149';

function drawRadar(pts) {
  const W = cv.width, H = cv.height, cx = W / 2, cy = H / 2;
  const R = Math.min(cx, cy) - 12;
  ctx.clearRect(0, 0, W, H);

  // 网格
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
  const rings = 4;
  for (let i = 1; i <= rings; i++) {
    ctx.beginPath(); ctx.arc(cx, cy, R * i / rings, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = '#484f58'; ctx.font = '11px system-ui';
    const d = (rangeM * i / rings).toFixed(0) + 'm';
    ctx.fillText(d, cx + 4, cy - R * i / rings + 12);
  }
  for (let a = 0; a < 360; a += 30) {
    const r = a * Math.PI / 180;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + R * Math.sin(r), cy - R * Math.cos(r));
    ctx.stroke();
  }
  // 角度标注
  ctx.fillStyle = '#6e7681'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  [['0°', 0, -1], ['90°', 1, 0], ['180°', 0, 1], ['270°', -1, 0]].forEach(([t, sx, sy]) => {
    ctx.fillText(t, cx + sx * (R + 2), cy + sy * (R + 2) + 4);
  });
  ctx.textAlign = 'start';

  // 扫描点
  if (!pts || !pts.length) return;
  for (const [deg, r] of pts) {
    if (r > rangeM) continue;
    const a = deg * Math.PI / 180;
    const rr = (r / rangeM) * R;
    const x = cx + rr * Math.sin(a), y = cy - rr * Math.cos(a);
    // 近=暖色, 远=冷色
    const t = Math.min(1, r / rangeM);
    const hue = 60 - 60 * t;             // 黄 -> 红
    ctx.fillStyle = `hsl(${185 - 140 * t}, 85%, ${62 - 12 * t}%)`;
    ctx.fillRect(x - psize / 2, y - psize / 2, psize, psize);
  }
  // 机器人本体
  ctx.fillStyle = '#58a6ff';
  ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill();
}

// ---- 雷达轮询 ----
let ldFrames = 0, ldT0 = performance.now(), lastId = -1;
async function pollScan() {
  try {
    const res = await fetch('/scan', { cache: 'no-store' });
    const d = await res.json();
    if (d.scan_id !== lastId) {
      lastId = d.scan_id;
      drawRadar(d.points);
      document.getElementById('ldPts').textContent = d.points.length;
      ldFrames++;
      const now = performance.now();
      if (now - ldT0 > 1000) {
        document.getElementById('ldFps').textContent =
          (ldFrames * 1000 / (now - ldT0)).toFixed(1) + ' Hz';
        ldFrames = 0; ldT0 = now;
        document.getElementById('ldDot').style.background = '#3fb950';
      }
    }
  } catch (e) {
    document.getElementById('ldDot').style.background = '#f85149';
  }
  setTimeout(pollScan, 40);
}
pollScan();
</script>
</body>
</html>
"""



# ------------------------------------------------------------------ 遥控 --

# 默认关闭串口遥控：webview 只负责相机 + 雷达。
# 串口必须由「真正跑策略的那一个进程」独占，两个进程同时开 /dev/ttyAMA0
# 会互相打断读取，表现为 HTTP 超时、姿态乱跳。
ENABLE_TELEOP = os.environ.get("LUWU_ENABLE_TELEOP") == "1"

TELEOP = {"vx": 0.0, "vyaw": 0.0, "t": 0.0}
TELEOP_TIMEOUT = 0.35        # 超过这么久没收到指令就自动停（失联保护）
VX_MAX, VYAW_MAX = 12.0, 40.0

_dog = None
_dog_lock = threading.Lock()
_teleop_started = False

# 可选：把 IMU 姿态角随 /scan 一起发出去，给上位机 SLAM 当旋转先验。
# 默认关闭——/dev/ttyAMA0 同一时刻只应有一个进程在读写；跑策略时不要再开这个。
ENABLE_IMU = os.environ.get("LUWU_ENABLE_IMU") == "1"
# 相机默认开；建图时设 LUWU_ENABLE_CAM=0 可以省下 ~1/3 CPU，让雷达发布率回到 10 Hz
ENABLE_CAM = os.environ.get("LUWU_ENABLE_CAM", "1") != "0"
IMU = {"r": 0.0, "p": 0.0, "y": 0.0, "t": 0.0, "ok": False}
_imu_lock = threading.Lock()
_imu_hist = deque(maxlen=600)      # 最近 ~30s 的姿态样本 (t, r, p, y)


def _imu_at(t):
    """取 t 时刻（或之前最近一次）的姿态，返回 (r, p, y, t)；没有就返回 None。"""
    with _imu_lock:
        hist = list(_imu_hist)
    if not hist:
        return None
    if hist[-1][0] <= t:
        return hist[-1]
    for item in reversed(hist):
        if item[0] <= t:
            return item
    return hist[0]


def _imu_json(sample):
    if not ENABLE_IMU or sample is None:
        return "null"
    t, r, p, y = sample
    return "[%.2f,%.2f,%.2f,%.3f]" % (r, p, y, t)


def _imu_loop():
    """20 Hz 读一次姿态角（与遥控共用同一个 XGO 实例，靠锁串行化）。"""
    while True:
        try:
            dog = _get_dog()
            with _dog_lock:
                r, p, y = dog.read_roll(), dog.read_pitch(), dog.read_yaw()
            with _imu_lock:
                t = time.time()
                IMU.update(r=r, p=p, y=y, t=t, ok=True)
                _imu_hist.append((t, r, p, y))
        except Exception:
            with _imu_lock:
                IMU["ok"] = False
        time.sleep(0.05)


def _get_dog():
    """懒加载 xgolib。launcher 停掉后 /dev/ttyAMA0 是空闲的。"""
    global _dog
    if _dog is None:
        from xgolib import XGO
        dog = XGO(port="/dev/ttyAMA0", version="auto")
        fw = dog.read_firmware()
        if fw in ("Null", None, ""):
            raise RuntimeError("连不上机器狗，检查 /dev/ttyAMA0 与狗是否上电")
        _dog = dog
        print("[teleop] 机器狗已连接，固件 %s" % fw, flush=True)
    return _dog


def _control_loop():
    """20 Hz 把当前速度指令写下去；指令过期就急停。"""
    applied = (0.0, 0.0)
    while True:
        with _dog_lock:
            vx, vyaw, ts = TELEOP["vx"], TELEOP["vyaw"], TELEOP["t"]
        if time.time() - ts > TELEOP_TIMEOUT:
            vx, vyaw = 0.0, 0.0
        if (vx, vyaw) != applied:
            try:
                dog = _get_dog()
                if vx == 0.0 and vyaw == 0.0:
                    dog.move_x(0)
                    dog.turn(0)
                    dog.stop()
                else:
                    dog.move_x(vx)
                    dog.turn(vyaw)
                applied = (vx, vyaw)
            except Exception as exc:
                print("[teleop] 下发失败: %s" % exc, flush=True)
                applied = (0.0, 0.0)
        time.sleep(0.05)


# ------------------------------------------------------------------ 路由 --

_cam = None
_lidar = None


def mjpeg_generator():
    """只推送新帧；客户端慢时自然丢帧，不会拖慢采集。"""
    if _cam is None:
        return
    last_seq = -1
    while True:
        jpg, seq, _fps = _cam.latest()
        if jpg is None or seq == last_seq:
            time.sleep(0.004)
            continue
        last_seq = seq
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
               + jpg + b"\r\n")


@app.route("/cmd")
def cmd():
    """vx: 前进/后退速度, vyaw: 转向角速度。两者都带失联保护。"""
    global _teleop_started
    if not ENABLE_TELEOP:
        return jsonify({"error": "teleop disabled; set LUWU_ENABLE_TELEOP=1 to enable"}), 403
    try:
        vx = float(request.args.get("vx", 0))
        vyaw = float(request.args.get("vyaw", 0))
    except ValueError:
        return jsonify({"error": "bad args"}), 400
    vx = max(-VX_MAX, min(VX_MAX, vx))
    vyaw = max(-VYAW_MAX, min(VYAW_MAX, vyaw))

    if not _teleop_started:
        _teleop_started = True
        threading.Thread(target=_control_loop, daemon=True).start()
        print("[teleop] 控制线程启动", flush=True)

    with _dog_lock:
        TELEOP.update(vx=vx, vyaw=vyaw, t=time.time())
    return jsonify({"vx": vx, "vyaw": vyaw})


@app.route("/stop")
def stop():
    if not ENABLE_TELEOP:
        return jsonify({"stopped": True, "note": "teleop disabled"})
    with _dog_lock:
        TELEOP.update(vx=0.0, vyaw=0.0, t=0.0)
    try:
        dog = _get_dog()
        dog.move_x(0)
        dog.turn(0)
        dog.stop()
    except Exception:
        pass
    return jsonify({"stopped": True})



@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/stream.mjpg")
def stream():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    if _cam is not None:
        _jpg, seq, fps = _cam.latest()
    else:
        seq, fps = 0, 0.0
    pts, sid, _t = _lidar.latest()
    return {"cam_fps": round(fps, 1), "cam_frames": seq,
            "lidar_hz_frames": sid, "lidar_points": len(pts)}


@app.route("/scan")
def scan():
    """兼容旧接口：只要最新一帧。"""
    pts, sid, t = _lidar.latest()
    return app.response_class(
        '{"scan_id":%d,"t":%.3f,"points":%s,"imu":%s}'
        % (sid, t, _pack(pts), _imu_json(_imu_at(t))),
        mimetype="application/json",
    )


@app.route("/frame_id")
def frame_id():
    """只要最新帧号（不带点云），给录包/订阅方做起始游标。"""
    with _lidar.frames_lock:
        if _lidar.frames:
            f = _lidar.frames[-1]
            return jsonify({"id": f["id"], "t": f["t"]})
    return jsonify({"id": -1, "t": 0.0})


@app.route("/imu")
def imu_samples():
    """IMU 全速率历史（20 Hz）：?since=<t>&max=<n>，单位秒。

    /frames 里的 imu 是"每帧一个采样"，做数据分析/录包请用这个接口拿完整流。
    """
    try:
        since = float(request.args.get("since", 0.0))
        mx = max(1, min(int(request.args.get("max", 100)), 500))
    except ValueError:
        return jsonify({"error": "bad params"}), 400
    with _imu_lock:
        samples = [s for s in _imu_hist if s[0] > since][:mx]
    body = ",".join("[%.3f,%.2f,%.2f,%.2f]" % tuple(s) for s in samples)
    return app.response_class('{"imu":[%s]}' % body, mimetype="application/json")


@app.route("/frames")
def frames():
    """批量取帧：?since=<id>&max=<n>

    每帧自带狗端采集时间戳和对应时刻的 IMU 姿态，客户端按 id 顺序消费即可，
    网络抖动不会再让"这一帧到底什么时候采的"变成未知数。
    """
    try:
        since = int(request.args.get("since", -1))
        mx = max(1, min(int(request.args.get("max", 20)), 60))
    except ValueError:
        return jsonify({"error": "bad params"}), 400
    with _lidar.frames_lock:
        fs = [f for f in _lidar.frames if f["id"] > since][:mx]
    body = ",".join(
        '{"id":%d,"t":%.3f,"pts":%s,"imu":%s}'
        % (f["id"], f["t"], _pack([(a, r) for a, r in f["pts"]]),
           _imu_json(_imu_at(f["t"])))
        for f in fs)
    return app.response_class('{"frames":[%s]}' % body, mimetype="application/json")


def _pack(pts):
    """把 [[deg,m],...] 压成紧凑 JSON，避免每帧几万字节的 key 开销。"""
    if not pts:
        return "[]"
    return "[" + ",".join("[%g,%g]" % (a, r) for a, r in pts) + "]"


def main():
    global _cam, _lidar

    # 顺序很重要：先 fork 雷达子进程，再初始化摄像头，
    # 这样子进程不会继承摄像头的文件描述符。
    print("[webview] 启动雷达子进程…")
    _lidar = LidarProc()
    if _lidar.wait_ready():
        print("[webview] 雷达 OK")
    else:
        print("[webview] 雷达无数据，页面上雷达区会是空的")

    if ENABLE_CAM:
        print("[webview] 启动摄像头…")
        _cam = Camera()
        print("[webview] 摄像头 OK")
    else:
        print("[webview] 相机已关闭（LUWU_ENABLE_CAM=0），CPU 让给雷达与建图")

    if ENABLE_IMU:
        print("[webview] 启用 IMU 随帧发布（LUWU_ENABLE_IMU=1）")
        threading.Thread(target=_imu_loop, daemon=True).start()

    print("[webview] 服务地址： http://%s:%d" % (_local_ip(), HTTP_PORT))
    app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True, debug=False,
            use_reloader=False)


def _local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _exit_on_sigterm(_signum, _frame):
    raise SystemExit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        main()
    except KeyboardInterrupt:
        print("\n[webview] 收到中断")
    finally:
        if _lidar:
            _lidar.close()
        if _cam:
            _cam.close()
