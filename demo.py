"""场景演示：两条传送带 + 机械臂键盘遥控。

运行：
    cd scene
    python demo.py              # 默认加载 show_scene.xml（展示版）
    python demo.py --scene train_scene.xml   # 最简训练版
    python demo.py --headless 5 # 不开窗口，空跑 5 秒做自检

键盘（每次按键步进一次，可连按）：
    A / D   底座左转 / 右转
    W / S   大臂上抬 / 下压
    Q / E   小臂收 / 伸
    Z / X   腕部上 / 下
    空格     夹爪开 / 合
    R       回到初始姿态

（机械臂底座已固定，没有滑轨平移了。）

传送带由 belts.apply_conveyors 驱动，见 belts.py。
"""

import argparse
import collections
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from belts import BELTS, apply_conveyors

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = np.array([-1.2, 0.20, -1.90, 1.20, 0.030, -0.030])
GRIPPER_OPEN = (0.030, -0.030)
GRIPPER_CLOSED = (0.000, 0.000)

# GLFW 键码
KEY_A, KEY_D = 65, 68
KEY_W, KEY_S = 87, 83
KEY_Q, KEY_E = 81, 69
KEY_Z, KEY_X = 90, 88
KEY_SPACE, KEY_R = 32, 82

# 每个按键对应 (ctrl 下标, 增量)：0=yaw, 1=shoulder, 2=elbow, 3=wrist
KEY_ACTIONS = {
    KEY_A: (0, -0.05), KEY_D: (0, +0.05),
    KEY_W: (1, -0.05), KEY_S: (1, +0.05),
    KEY_Q: (2, -0.05), KEY_E: (2, +0.05),
    KEY_Z: (3, -0.05), KEY_X: (3, +0.05),
}


def clip_ctrl(model, data):
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        data.ctrl[i] = np.clip(data.ctrl[i], lo, hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", type=float, default=0.0,
                        help="不打开窗口，按给定秒数空跑")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="传送带速度倍率")
    parser.add_argument("--scene", default="show_scene.xml",
                        help="要加载的模型文件：show_scene.xml（默认）或 train_scene.xml")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(os.path.join(HERE, args.scene))
    data = mujoco.MjData(model)

    # 传送带速度倍率
    belts = [dict(b, speed=b["speed"] * args.speed) for b in BELTS]

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = HOME
    mujoco.mj_forward(model, data)

    if args.headless > 0:
        n = int(args.headless / model.opt.timestep)
        for _ in range(n):
            apply_conveyors(model, data, belts)
            mujoco.mj_step(model, data)
        print("headless OK")
        for i in range(3):
            print(f"  cube_{i}: {np.round(data.xpos[model.body(f'cube_{i}').id], 3)}")
        print("  nan:", bool(np.isnan(data.qacc).any()))
        return

    pending = collections.deque()
    state = {"gripper_open": True}

    def on_key(keycode):
        pending.append(int(keycode))

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.show_left_ui = False
        viewer.show_right_ui = False
        
        viewer.cam.distance = 1.9
        viewer.cam.elevation = -28
        viewer.cam.azimuth = 125
        viewer.cam.lookat[:] = [0.15, 0.10, 0.30]

        frame_dt = 0.01
        print("窗口已打开。键盘控制见 demo.py 顶部说明，Esc 退出。")

        while viewer.is_running():
            t0 = time.perf_counter()

            # 处理这一帧收到的按键
            while pending:
                key = pending.popleft()
                if key == KEY_SPACE:
                    state["gripper_open"] = not state["gripper_open"]
                    pair = GRIPPER_OPEN if state["gripper_open"] else GRIPPER_CLOSED
                    data.ctrl[4], data.ctrl[5] = pair
                elif key == KEY_R:
                    data.ctrl[:] = HOME
                    state["gripper_open"] = True
                elif key in KEY_ACTIONS:
                    idx, delta = KEY_ACTIONS[key]
                    data.ctrl[idx] += delta
                    clip_ctrl(model, data)

            apply_conveyors(model, data, belts)
            for _ in range(int(frame_dt / model.opt.timestep)):
                mujoco.mj_step(model, data)

            viewer.sync()
            rest = frame_dt - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)


if __name__ == "__main__":
    main()
