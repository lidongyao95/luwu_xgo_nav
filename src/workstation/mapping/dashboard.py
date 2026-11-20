#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGO 机器狗 —— 键盘遥控 + 建图 一体化驾驶台

一个网页里同时看三样东西：
    左上  相机实时画面（经本机代理，帧率不用高）
    左下  雷达实时扫描
    右    正在构建的 2D 栅格地图 + 轨迹

键盘（点击页面后生效）：
    ↑ / W      前进
    ↓ / S      后退
    ← / A      左转
    → / D      右转
    空格        立即停车
松开按键 0.35 秒内狗端会自动急停（狗端看门狗），关掉网页也会停。

依赖：狗上跑着 webview.py（提供 /stream.mjpg /scan /cmd）
      python3 dashboard.py --host 172.18.11.103
然后浏览器打开 http://127.0.0.1:8010
"""

import argparse
from pathlib import Path
import sys
import threading
import time

import requests
from flask import Flask, Response, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "slam"))
import slam2d

APP = Flask(__name__)

CFG = {"dog": "172.18.11.103:8000", "cmd": "172.18.11.103:8100"}

_slam = None
_source = None


# ------------------------------------------------------------------ 代理 --

def _dog_url(path):
    return "http://%s%s" % (CFG["dog"], path)


@APP.route("/cam.mjpg")
def cam():
    """把狗端的 MJPEG 原样转发，浏览器只跟本机一个源打交道。"""
    try:
        r = requests.get(_dog_url("/stream.mjpg"), stream=True, timeout=10)
    except Exception:
        return Response("camera offline", status=503)

    def gen():
        try:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            r.close()

    return Response(gen(), mimetype=r.headers.get("Content-Type",
                                                   "multipart/x-mixed-replace; boundary=frame"))


@APP.route("/scan")
def scan():
    try:
        r = requests.get(_dog_url("/scan"), timeout=2)
        return Response(r.content, mimetype="application/json",
                        headers={"Cache-Control": "no-store"})
    except Exception:
        return jsonify({"scan_id": -1, "points": []})


@APP.route("/cmd")
def cmd():
    try:
        vx = float(request.args.get("vx", 0))
        vyaw = float(request.args.get("vyaw", 0))
    except ValueError:
        return jsonify({"error": "bad args"}), 400
    try:
        requests.get("http://%s/cmd" % CFG["cmd"],
                     params={"vx": vx, "vyaw": vyaw}, timeout=1.5)
        return jsonify({"vx": vx, "vyaw": vyaw})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@APP.route("/stop")
def stop():
    try:
        requests.get("http://%s/stop" % CFG["cmd"], timeout=1.5)
    except Exception:
        pass
    return jsonify({"stopped": True})


# ------------------------------------------------------------------ 本机 --

@APP.route("/map.png")
def map_png():
    return Response(_slam.map_png(blank_size=200), mimetype="image/png",
                    headers={"Cache-Control": "no-store"})


@APP.route("/stats")
def stats():
    state = _slam.stats()
    try:
        dog = requests.get(_dog_url("/stats"), timeout=1.5).json()
    except Exception:
        dog = {}
    return jsonify({
        **state,
        "lidar_hz": _source.hz,
        "cam_fps": dog.get("cam_fps", 0),
    })


PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>XGO · 遥控建图</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;background:#0d1117;color:#c9d1d9;font:13px/1.5 -apple-system,"PingFang SC",sans-serif;
      height:100vh;display:flex;flex-direction:column;overflow:hidden}
 header{display:flex;gap:18px;align-items:center;padding:8px 16px;
        border-bottom:1px solid #21262d;flex:none}
 h1{font-size:14px;margin:0;font-weight:600;letter-spacing:.5px}
 .s{font-variant-numeric:tabular-nums;color:#8b949e}
 .s b{color:#58a6ff;font-weight:600}
 .keys{margin-left:auto;color:#6e7681;font-size:12px}
 .keys kbd{background:#161b22;border:1px solid #30363d;border-radius:3px;
           padding:1px 5px;font-size:11px;font-family:inherit}
 main{flex:1;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1.15fr 1fr;
      gap:1px;background:#21262d;min-height:0}
 .pane{background:#0d1117;display:flex;flex-direction:column;min-height:0;min-width:0}
 .pane h2{margin:0;padding:5px 12px;font-size:11px;font-weight:500;color:#8b949e;
          border-bottom:1px solid #21262d;flex:none;letter-spacing:.4px}
 .body{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;padding:8px;min-height:0}
 .map{grid-row:1 / span 2;grid-column:2}
 img{max-width:100%;max-height:100%;display:block}
 img.raster{image-rendering:pixelated;border-radius:3px}
 canvas{max-width:100%;max-height:100%}
 #err{color:#f85149;font-size:12px}
</style></head><body>
<header>
 <h1>XGO · 遥控建图</h1>
 <span class="s">位姿 <b id="pose">—</b></span>
 <span class="s">帧 <b id="frames">0</b></span>
 <span class="s">地图 <b id="area">—</b></span>
 <span class="s">雷达 <b id="hz">—</b></span>
 <span class="s">相机 <b id="cfps">—</b></span>
 <span class="keys"><kbd>W</kbd>/<kbd>S</kbd> 前后 · <kbd>A</kbd>/<kbd>D</kbd> 转向 · <kbd>空格</kbd> 停</span>
</header>
<main>
 <section class="pane">
   <h2>CAMERA</h2>
   <div class="body"><img id="cam" src="/cam.mjpg"></div>
 </section>
 <section class="pane map">
   <h2>MAP · scan matching</h2>
   <div class="body"><img id="map" class="raster" src="/map.png"></div>
 </section>
 <section class="pane">
   <h2>LIDAR</h2>
   <div class="body"><canvas id="radar" width="520" height="520"></canvas></div>
 </section>
</main>
<script>
// ---------------- 键盘遥控 ----------------
const held = new Set();
const VX = 0.35, VYAW = 0.8;          // 策略单位: m/s, rad/s
let lastSent = 0;

function cmdFromKeys(){
  let vx = 0, vyaw = 0;
  if (held.has('w') || held.has('arrowup'))    vx += VX;
  if (held.has('s') || held.has('arrowdown'))  vx -= VX;
  if (held.has('a') || held.has('arrowleft'))  vyaw += VYAW;
  if (held.has('d') || held.has('arrowright')) vyaw -= VYAW;
  return {vx, vyaw};
}
async function sendCmd(force){
  const {vx, vyaw} = cmdFromKeys();
  const now = Date.now();
  if (!force && now - lastSent < 80) return;
  lastSent = now;
  try {
    await fetch(`/cmd?vx=${vx}&vyaw=${vyaw}`, {cache:'no-store'});
    document.getElementById('err')?.remove();
  } catch(e) {}
}
addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (['arrowup','arrowdown','arrowleft','arrowright',' '].includes(k)) e.preventDefault();
  if (k === ' ') { fetch('/stop'); held.clear(); return; }
  if (!held.has(k)) { held.add(k); sendCmd(true); }
});
addEventListener('keyup', e => {
  const k = e.key.toLowerCase();
  held.delete(k);
  sendCmd(true);
});
addEventListener('blur', () => { held.clear(); fetch('/stop'); });
// 按住不放时持续以 ~12 Hz 发指令，狗端看门狗 0.35 s
setInterval(() => { if (held.size) sendCmd(false); }, 80);

// ---------------- 雷达绘制 ----------------
const cv = document.getElementById('radar'), ctx = cv.getContext('2d');
const RANGE = 6.0;
function drawRadar(pts){
  const W = cv.width, H = cv.height, cx = W/2, cy = H/2, R = Math.min(cx,cy)-10;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle = '#21262d';
  for (let i=1;i<=4;i++){ ctx.beginPath(); ctx.arc(cx,cy,R*i/4,0,Math.PI*2); ctx.stroke(); }
  for (let a=0;a<360;a+=30){
    const r=a*Math.PI/180;
    ctx.beginPath(); ctx.moveTo(cx,cy);
    ctx.lineTo(cx+R*Math.sin(r), cy-R*Math.cos(r)); ctx.stroke();
  }
  for (const [deg, r] of pts){
    if (r > RANGE) continue;
    const a = deg*Math.PI/180, rr = r/RANGE*R;
    const t = Math.min(1, r/RANGE);
    ctx.fillStyle = `hsl(${185-140*t},85%,${62-12*t}%)`;
    ctx.fillRect(cx+rr*Math.sin(a)-1, cy-rr*Math.cos(a)-1, 2, 2);
  }
  ctx.fillStyle='#58a6ff';
  ctx.beginPath(); ctx.arc(cx,cy,5,0,Math.PI*2); ctx.fill();
}

// ---------------- 主循环 ----------------
let lastScan = -1, lastFrames = -1, camT = Date.now(), camN = 0;
const camImg = document.getElementById('cam');
camImg.onload = () => { camN++; };
setInterval(() => {
  const now = Date.now();
  document.getElementById('cfps').textContent = (camN*1000/(now-camT)).toFixed(1)+' fps';
  camN = 0; camT = now;
}, 1000);

async function tick(){
  try {
    const d = await (await fetch('/stats',{cache:'no-store'})).json();
    document.getElementById('pose').textContent =
      `x=${d.x.toFixed(2)} y=${d.y.toFixed(2)} θ=${(d.th*180/Math.PI).toFixed(0)}°`;
    document.getElementById('frames').textContent = d.frames;
    document.getElementById('area').textContent = d.known_m2.toFixed(1)+' m²';
    document.getElementById('hz').textContent = (d.lidar_hz||0).toFixed(1)+' Hz';
    if (d.frames !== lastFrames){
      lastFrames = d.frames;
      document.getElementById('map').src = '/map.png?t='+Date.now();
    }
    const s = await (await fetch('/scan',{cache:'no-store'})).json();
    if (s.scan_id !== lastScan){ lastScan = s.scan_id; drawRadar(s.points); }
  } catch(e) {}
  setTimeout(tick, 80);
}
tick();
</script></body></html>"""


@APP.route("/")
def index():
    return PAGE


def slam_loop():
    while True:
        if not _slam.step():
            time.sleep(0.01)


def main():
    global _slam, _source
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.18.11.103")
    ap.add_argument("--dog-port", type=int, default=8000)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--cmd-port", type=int, default=8100, help="狗端策略指令接口端口")
    args = ap.parse_args()

    CFG["dog"] = "%s:%d" % (args.host, args.dog_port)
    CFG["cmd"] = "%s:%d" % (args.host, args.cmd_port)
    print("[dash] 狗端 %s" % CFG["dog"])

    _source = slam2d.ScanSource(args.host, args.dog_port)
    time.sleep(1.5)
    _slam = slam2d.SLAM(_source)

    threading.Thread(target=slam_loop, daemon=True).start()

    print("[dash] 打开 http://127.0.0.1:%d" % args.port)
    APP.run(host="127.0.0.1", port=args.port, threaded=True,
            debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
