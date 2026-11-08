"""传送带驱动。

MuJoCo 没有原生的"传送带表面"。这里的做法是：
检测与带面 geom 接触的自由刚体，沿皮带运动方向做一个速度伺服
（等价于皮带对物块的摩擦力），并限制加速度，避免把物块弹飞。

用法：
    from belts import apply_conveyors
    apply_conveyors(model, data)
    mujoco.mj_step(model, data)
"""

import numpy as np
import mujoco

# 与 scene.xml 中的带面 geom 对应；direction 是世界坐标下的输送方向
BELTS = [
    {"surface": "beltA_surface", "direction": (1.0, 0.0, 0.0), "speed": 0.15},
    {"surface": "beltB_surface", "direction": (0.0, -1.0, 0.0), "speed": 0.12},
]


def apply_conveyors(model, data, belts=None, gain=200.0, max_accel=40.0):
    """给传送带上的自由刚体施加沿带方向的驱动力。"""
    belts = BELTS if belts is None else belts
    data.xfrc_applied[:] = 0.0

    for belt in belts:
        gid = model.geom(belt["surface"]).id
        direction = np.asarray(belt["direction"], dtype=float)
        direction /= np.linalg.norm(direction)
        speed = float(belt["speed"])

        driven = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 == gid:
                other = contact.geom2
            elif contact.geom2 == gid:
                other = contact.geom1
            else:
                continue

            body = int(model.geom_bodyid[other])
            if body == 0 or body in driven:
                continue
            if model.body_jntnum[body] == 0:
                continue
            joint = int(model.body_jntadr[body])
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                continue

            driven.add(body)
            dof = int(model.jnt_dofadr[joint])
            v_along = float(np.dot(data.qvel[dof:dof + 3], direction))
            accel = float(np.clip(gain * (speed - v_along), -max_accel, max_accel))
            data.xfrc_applied[body, :3] += model.body_mass[body] * accel * direction
