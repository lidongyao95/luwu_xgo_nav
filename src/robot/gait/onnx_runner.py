#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGO 真机部署 —— 用 luwu_mjlab 训出来的 Mini2S-Walk 策略跑推理

策略接口（从 policy.onnx 元数据读出，已核对）：
    obs [1,56] → actions [1,15]
    obs  = base_ang_vel(3), projected_gravity(3), command(3),
           phase(2), joint_pos(15), joint_vel(15), actions(15)
    action_scale = 0.25
    default_joint_pos = 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,-1.570,1.350   (弧度)
    控制频率 = 40 Hz（仿真 dt 0.005 × decimation 5）

用法：
    python3 onnx_runner.py --dry             # 只算不发动，打印观测/动作
    python3 onnx_runner.py --probe 31 10     # 单独把 31 号舵机转 +10°，用于核对关节映射
    python3 onnx_runner.py --run             # 闭环跑策略（⚠️ 会让狗动）

已知的两个妥协（必须清楚）：
  1. 这个固件（M-6.0.2）上 dog.read_imu() 返回的是死数据（每次完全相同，
     含 2.99e36 这种垃圾值），只有 read_roll/pitch/yaw 是活的。
     所以 base_ang_vel 只能用姿态角差分 + 低通估计，噪声偏大。
  2. 手臂三个关节在真机上收在 (-65, 50, -75)°，和仿真默认 (0, -90, 77)° 完全不同。
     手臂总质量只有 ~40 g，对动力学影响可忽略，所以这里**不驱动手臂**，
     观测里也把手臂按仿真默认值（相对量 0）上报，避免给策略喂分布外的输入。
"""

import argparse
import json
import math
import os
import time

import numpy as np
import onnxruntime as ort

# 优先用与本脚本同目录的策略文件（目录整理后保持自包含），
# 找不到再回退到狗上的部署路径。
POLICY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "mini2s_walk_policy.onnx")
if not os.path.exists(POLICY):
    POLICY = "/home/pi/Chapter4/mini2s_walk_policy.onnx"
CTRL_HZ = 40.0

# ONNX 元数据里的关节顺序（15 个）
JOINT_NAMES = [
    "fl_hip", "fl_thigh", "fl_calf",
    "fr_hip", "fr_thigh", "fr_calf",
    "br_hip", "br_thigh", "br_calf",
    "bl_hip", "bl_thigh", "bl_calf",
    "arm_yaw", "arm_thigh", "arm_calf",
]
DEFAULT_POS = np.deg2rad([0, 57.3, 0, 0, 57.3, 0, 0, 57.3, 0, 0, 57.3, 0,
                          0, -90.0, 77.4])
ACTION_SCALE = 0.25

# XGO 舵机号：每条腿 3 个（hip, thigh, calf）
# 顺序按 xgolib 的 motor() MOTOR_ID，但 br/bl 是否对应 31/41 需实测确认
LEG_SERVO_IDS = [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43]
ARM_SERVO_IDS = [51, 52, 53]

TILT_ABORT_DEG = 25.0
ANG_VEL_ALPHA = 0.12      # 姿态差分估计角速度的低通系数，越小越平滑
SLEW_DEG = 4.0            # 每个控制周期允许的最大关节变化（度）→ 4°×40Hz = 160°/s
CMD = {"vx": 0.0, "wz": 0.0, "t": 0.0}   # 策略单位：m/s, rad/s
CMD_TIMEOUT = 0.5                        # 超时回零，和键盘失联保护一致
PHASE_PERIOD = 0.4
CMD_DEADBAND = 0.1


# ----------------------------------------------------------- 观测组装 --

class Imu:
    """read_imu() 在这台狗上是坏的，只能用姿态角 + 差分。"""

    def __init__(self, dog, alpha=ANG_VEL_ALPHA):
        self.dog = dog
        self.alpha = alpha
        self.prev = None
        self.prev_t = None
        self.omega = np.zeros(3, dtype=np.float32)

    def read_attitude(self):
        """返回 (roll, pitch, yaw) 弧度。"""
        r = math.radians(self.dog.read_roll())
        p = math.radians(self.dog.read_pitch())
        y = math.radians(self.dog.read_yaw())
        return r, p, y

    def update(self, dt):
        r, p, y = self.read_attitude()
        now = time.time()
        if self.prev is not None and dt > 0:
            dr = _wrap(r - self.prev[0]) / dt
            dp = _wrap(p - self.prev[1]) / dt
            dy = (y - self.prev[2]) / dt
            # 欧拉角速度 → 机体角速度（ZYX 欧拉）
            sr, cr = math.sin(r), math.cos(r)
            sp, cp = math.sin(p), math.cos(p)
            wx = dr - dy * sp
            wy = dp * cr + dy * sr * cp
            wz = -dp * sr + dy * cr * cp
            raw = np.array([wx, wy, wz], dtype=np.float32)
            self.omega = self.alpha * raw + (1 - self.alpha) * self.omega
        self.prev = (r, p, y)
        self.prev_t = now
        return r, p, y

    @staticmethod
    def projected_gravity(r, p):
        """机体系下的重力方向 R^T·[0,0,-1]（ZYX 欧拉）。"""
        return np.array([math.sin(p),
                         -math.sin(r) * math.cos(p),
                         -math.cos(r) * math.cos(p)], dtype=np.float32)


def _wrap(a):
    """把角度差折到 [-pi, pi]。用取模，避免遇到异常大值时死循环。"""
    if not math.isfinite(a):
        return 0.0
    return (a + math.pi) % (2 * math.pi) - math.pi


class PolicyRunner:
    def __init__(self, policy_path=POLICY, verbose=True):
        self.sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
        self.action = np.zeros(15, dtype=np.float32)
        self.t0 = time.time()

        if verbose:
            names, _ = self._meta(policy_path)
            print("[runner] 策略加载完成")
            print("[runner] 关节顺序:", ",".join(names) if names else JOINT_NAMES)

    @staticmethod
    def _meta(path):
        try:
            import onnx
            m = onnx.load(path)
            d = {kv.key: kv.value for kv in m.metadata_props}
            names = d.get("joint_names", "").split(",")
            return names, d
        except Exception:
            return [], {}

    def reset_clock(self):
        self.t0 = time.time()

    def phase(self, t, cmd):
        frac = (t % PHASE_PERIOD) / PHASE_PERIOD
        s, c = math.sin(frac * 2 * math.pi), math.cos(frac * 2 * math.pi)
        if float(np.linalg.norm(cmd)) < CMD_DEADBAND:
            return np.zeros(2, dtype=np.float32)
        return np.array([s, c], dtype=np.float32)

    def infer(self, obs):
        self.action = self.sess.run(None, {self.inp: obs[None, :]})[0][0].astype(np.float32)
        return self.action


def build_obs(imu, joints_rad, joint_vel, cmd, policy, t, action_prev):
    """按 ONNX 元数据显示的顺序拼 56 维。"""
    r, p, _ = imu.prev
    parts = [
        imu.omega,                              # base_ang_vel 3
        Imu.projected_gravity(r, p),            # projected_gravity 3
        np.asarray(cmd, dtype=np.float32),      # command 3
        policy.phase(t, cmd),                   # phase 2
        joints_rad - DEFAULT_POS,               # joint_pos 15 (相对)
        joint_vel,                              # joint_vel 15
        action_prev,                            # actions 15
    ]
    obs = np.concatenate(parts).astype(np.float32)
    # 串口偶尔读失败会带出 NaN/Inf，先消毒再喂网络
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------- 主流程 --

class Robot:
    def __init__(self):
        from xgolib import XGO
        self.dog = XGO(port="/dev/ttyAMA0", version="auto")
        fw = self.dog.read_firmware()
        if fw in ("Null", None, ""):
            raise RuntimeError("连不上机器狗")
        print("[robot] 固件", fw)

    def read_joints(self):
        """读 15 个关节角（度）→ 弧度。读不到的按默认值补。"""
        q = self.dog.read_motor()
        if len(q) < 15:
            print("[robot] 警告：read_motor 只返回 %d 个值" % len(q))
            q = list(q) + [0.0] * (15 - len(q))
        q = np.deg2rad(np.asarray(q[:15], dtype=np.float64))
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        # 手臂不参与控制，按仿真默认值上报
        q[12:15] = DEFAULT_POS[12:15]
        return q

    def send_legs(self, q_rad):
        """只下发 12 条腿；手臂保持它自己的姿态不动。"""
        degs = np.rad2deg(q_rad[:12])
        self.dog.motor(LEG_SERVO_IDS, [float(v) for v in degs])

    def stop(self):
        try:
            self.dog.move_x(0)
            self.dog.turn(0)
            self.dog.stop()
        except Exception:
            pass


def run_dry(robot, policy, seconds=3.0):
    imu = Imu(robot.dog)
    dt = 1.0 / CTRL_HZ
    q_prev = robot.read_joints()
    imu.update(dt)
    policy.reset_clock()
    t_start = time.time()
    n = 0
    lat = []
    while time.time() - t_start < seconds:
        loop0 = time.time()
        imu.update(dt)
        q = robot.read_joints()
        jvel = np.zeros(15, dtype=np.float32)
        jvel[:12] = ((q - q_prev)[:12] / dt).astype(np.float32)
        q_prev = q

        cmd = np.zeros(3, dtype=np.float32)
        obs = build_obs(imu, q, jvel, cmd, policy, time.time() - t_start, policy.action)
        t_inf = time.time()
        act = policy.infer(obs)
        lat.append((time.time() - t_inf) * 1000)

        q_target = DEFAULT_POS + ACTION_SCALE * act
        n += 1
        if n % 10 == 0:
            print("[dry] t=%5.2fs obs[:9]=%s" % (time.time() - t_start,
                                                 np.round(obs[:9], 3)))
            print("       act[:6]=%s  目标(度)=%s" % (
                np.round(act[:6], 3), np.round(np.rad2deg(q_target[:6]), 1)))
        time.sleep(max(0.0, dt - (time.time() - loop0)))

    print("\n[dry] 共 %d 帧, 实际 %.1f Hz, ONNX 平均 %.2f ms"
          % (n, n / (time.time() - t_start), float(np.mean(lat))))
    print("[dry] 未向舵机下发任何指令")


def run_probe_group(robot, first_id, deg, hold=1.8):
    """整条腿（hip/thigh/calf）一起转，动作更明显。"""
    q = robot.read_joints()
    if first_id not in LEG_SERVO_IDS:
        print("支持的腿起始舵机号: 11 / 21 / 31 / 41")
        return
    i0 = LEG_SERVO_IDS.index(first_id)
    print("[probe-group] 舵机组 %d-%d 抖动 %+.1f°（保持 %.1fs）"
          % (first_id, first_id + 2, deg, hold))
    target = q.copy()
    for k in range(3):
        target[i0 + k] = q[i0 + k] + math.radians(deg if k != 2 else -deg)
    robot.send_legs(target)
    try:
        time.sleep(hold)
    finally:
        robot.send_legs(q)
        print("[probe-group] 已复位")


def run_probe(robot, servo_id, deg, hold=1.5):
    """把单个舵机转 deg 度、保持 hold 秒、然后自动复位。

    用来核对「MJCF 关节顺序」和「XGO 舵机编号」的对应关系：
    看哪条腿动了，就知道 31-33 到底是左后还是右后。
    """
    q = robot.read_joints()
    if servo_id not in LEG_SERVO_IDS:
        print("支持的舵机号:", LEG_SERVO_IDS)
        return
    idx = LEG_SERVO_IDS.index(servo_id)
    before = float(np.rad2deg(q[idx]))
    print("[probe] %d 号舵机 当前 %.1f° → %.1f°（保持 %.1fs 后自动复位）"
          % (servo_id, before, before + deg, hold))

    target = q.copy()
    target[idx] = q[idx] + math.radians(deg)
    robot.send_legs(target)
    try:
        time.sleep(hold)
    finally:
        robot.send_legs(q)
        print("[probe] 已复位")


def start_cmd_server(port):
    """开一个小 HTTP 接口，让本机键盘遥控驱动策略。"""
    from flask import Flask, jsonify, request

    app = Flask("cmdsrv")

    @app.route("/cmd")
    def _cmd():
        CMD["vx"] = max(-0.8, min(0.8, float(request.args.get("vx", 0))))
        CMD["wz"] = max(-1.5, min(1.5, float(request.args.get("vyaw", 0))))
        CMD["t"] = time.time()
        return jsonify(CMD)

    @app.route("/stop")
    def _stop():
        CMD.update(vx=0.0, wz=0.0, t=0.0)
        return jsonify({"stopped": True})

    print("[cmdsrv] 指令接口 http://0.0.0.0:%d/cmd?vx=..&vyaw=.." % port)
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False, use_reloader=False)


def current_cmd():
    if time.time() - CMD["t"] > CMD_TIMEOUT:
        return np.zeros(3, dtype=np.float32)
    return np.array([CMD["vx"], 0.0, CMD["wz"]], dtype=np.float32)


def run_closed_loop(robot, policy, seconds=10.0):
    """⚠️ 会让狗真的动。指令来自 /cmd，超时自动回零。"""
    imu = Imu(robot.dog)
    dt = 1.0 / CTRL_HZ
    q_prev = robot.read_joints()
    imu.update(dt)
    policy.reset_clock()

    print("[run] 3 秒后开始，先保持零指令…… Ctrl+C 停")
    time.sleep(3.0)

    # 从当前实际姿态起步，避免第一帧就跳变
    q_cmd = robot.read_joints().copy()
    max_step = math.radians(SLEW_DEG)

    t_start = time.time()
    n = 0
    try:
        while time.time() - t_start < seconds:
            loop0 = time.time()
            imu.update(dt)
            q = robot.read_joints()
            jvel = np.zeros(15, dtype=np.float32)
            jvel[:12] = ((q - q_prev)[:12] / dt).astype(np.float32)
            q_prev = q

            cmd = current_cmd()
            obs = build_obs(imu, q, jvel, cmd, policy, time.time() - t_start, policy.action)
            act = policy.infer(obs)

            if not np.isfinite(act).all():
                print("[run] ⚠️ 策略输出 NaN/Inf，跳过该帧")
                q_prev = robot.read_joints()
                continue
            q_target = DEFAULT_POS + ACTION_SCALE * act
            # 安全钳位：任何关节超出 ±90° 都不可信
            q_target = np.clip(q_target, -math.pi / 2, math.pi / 2)
            # 速率限制：每周期最多变 SLEW_DEG，防止猛甩
            q_target = np.clip(q_target, q_cmd - max_step, q_cmd + max_step)
            q_cmd = q_target
            robot.send_legs(q_target)

            n += 1
            r, p, _ = imu.prev
            # 安全保护：倾角过大立刻停
            if abs(math.degrees(r)) > TILT_ABORT_DEG or abs(math.degrees(p)) > TILT_ABORT_DEG:
                print("[run] ⚠️ 倾角超限 roll=%.1f° pitch=%.1f°，急停"
                      % (math.degrees(r), math.degrees(p)))
                break
            if n % 20 == 0:
                print("[run] t=%4.1fs roll=%+5.1f° pitch=%+5.1f° cmd=[%.2f %.2f]"
                      % (time.time() - t_start, math.degrees(r), math.degrees(p),
                         cmd[0], cmd[2]))
            time.sleep(max(0.0, dt - (time.time() - loop0)))
    finally:
        robot.stop()
        print("[run] 停止，共 %d 帧" % n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只计算不发动")
    ap.add_argument("--run", action="store_true", help="闭环跑策略（会动）")
    ap.add_argument("--probe", nargs=2, metavar=("SERVO_ID", "DEG"),
                    help="单独转一个舵机，用于核对关节映射")
    ap.add_argument("--probe-group", nargs=2, metavar=("FIRST_SERVO_ID", "DEG"),
                    help="整条腿三个关节一起转，更容易看清是哪条腿")
    ap.add_argument("--seconds", type=float, default=600.0,
                    help="闭环运行时长，默认 10 分钟")
    ap.add_argument("--serve-cmd", type=int, default=0,
                    help="在该端口开 /cmd 指令接口，供本机键盘遥控")
    ap.add_argument("--policy", default=POLICY)
    args = ap.parse_args()

    if not os.path.exists(args.policy):
        raise SystemExit("找不到策略文件: " + args.policy)

    robot = Robot()
    policy = PolicyRunner(args.policy)

    try:
        if args.dry:
            run_dry(robot, policy, args.seconds)
        elif args.probe:
            run_probe(robot, int(args.probe[0]), float(args.probe[1]))
        elif args.probe_group:
            run_probe_group(robot, int(args.probe_group[0]), float(args.probe_group[1]))
        elif args.run:
            if args.serve_cmd:
                import threading
                threading.Thread(target=start_cmd_server, args=(args.serve_cmd,),
                                 daemon=True).start()
                time.sleep(1.0)
            run_closed_loop(robot, policy, args.seconds)
        else:
            print("请指定 --dry / --probe / --run 之一")
    except KeyboardInterrupt:
        print("\n[main] 中断")
    finally:
        robot.stop()
        try:
            robot.dog.ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
