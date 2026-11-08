import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    def __init__(self,verbose=0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps < 500000 :
            self.training_env.env_method("_set_dist",0.05)
            self.training_env.env_method("_set_angle",0.9)
        elif self.num_timesteps < 1000000 :
            self.training_env.env_method("_set_dist",0.04)
            self.training_env.env_method("_set_angle",0.97)
        else :
            self.training_env.env_method("_set_dist",0.03)
            self.training_env.env_method("_set_angle",0.99)

        return True

class ArmEnv(gym.Env):
    def __init__(self):
        super().__init__()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xml_path = os.path.join(base_dir, "assets", "models", "train_scene.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1,high=1.0,shape=(6,),dtype=np.float32)
        self.ctrl_scale = np.array([3.0,3.0,3.0,3.0,3.14,0.03,-0.03],dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-np.inf,high=np.inf,shape=(22,),dtype=np.float32
        )

        self.dist_threshold = 0.05
        self.angle_threshold = 0.9

        self.steps = 0
        self.max_steps = 150
        self.phase = 1

        self.arm_joints = [
            self.model.joint(n).id
            for n in ("joint_yaw", "joint_shoulder", "joint_elbow", "joint_wrist")
        ]
        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joints]
        self.arm_dof_adr = self.model.jnt_dofadr[self.arm_joints]

        self.box_pos = np.zeros(3)
        self.box_body_id = self.model.body("cube_0").id
        self.box_geom_id = self.model.geom("cube_0_geom").id
        self.mocap_id = self.model.body_mocapid[self.box_body_id]

        self.prev_action = np.zeros(6, dtype=np.float64)

        self.ee_id = self.model.site("ee_site").id
        self.palm_id = self.model.geom("palm").id
        self.finger_joint_id = self.model.joint("joint_finger_L").id
        self.finger_qpos_adr = self.model.jnt_qposadr[self.finger_joint_id]
        
    def reset(self,seed=None,option=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model,self.data)
        
        #随机生成方块
        box_x = self.np_random.uniform(-0.50+0.025,0.3-0.025)
        box_y = self.np_random.uniform(0.4+0.025,0.9-0.025)
        self.box_pos = np.array([box_x,box_y,0.35+0.025])
        self.data.mocap_pos[self.mocap_id] = self.box_pos
        self.data.mocap_quat[self.mocap_id] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model,self.data)

        self.steps = 0
        self.phase = 1
        self.prev_action = np.zeros(6, dtype=np.float64)
        return self._get_obs(),{}
        """
        return obs,
        obs:姿态
        """

    def step(self,action):
        action = np.clip(np.asarray(action,dtype=np.float64),-1.0,1.0)
        
        self.data.ctrl[:5] = action[:5] * self.ctrl_scale[:5]
        gripper_cmd = (action[5] + 1.0) / 2.0 * 0.03
        self.data.ctrl[5] = gripper_cmd
        self.data.ctrl[6] = -gripper_cmd

        for _ in range(5) :
            mujoco.mj_step(self.model,self.data)
        
        self.steps += 1

        if np.isnan(self.data.qacc).any():
            return self._get_obs(), -10.0, True, False, {"nan": True}

        info={}
        truncated = self.steps >= self.max_steps

        reward = 0
        if self.phase == 1 :
            joint_3_pos = self.data.xanchor[self.arm_joints[3]].copy()
            target_pos = np.array([self.box_pos[0],self.box_pos[1],self.box_pos[2]+0.25])
            rel = joint_3_pos-target_pos
            dist = float(np.linalg.norm(rel))
            ee_pos = self.data.site_xpos[self.ee_id]
            forward = ee_pos-joint_3_pos
            forward = forward/np.linalg.norm(forward)
            angle = float(np.dot(forward,np.array([0,0,-1])))
            angle_error = 1.0 - angle
            gripper_open_value = float(self.data.qpos[self.finger_qpos_adr])

            reward -= dist*2
            reward -= (angle_error ** 2) * 10.0
            if (angle>0.98) :reward+=0.1

            reward += gripper_open_value

            action_change = np.sum(np.square(action - self.prev_action))
            reward -= action_change * 0.5
            self.prev_action = action.copy()

            if (dist < self.dist_threshold and angle>self.angle_threshold) :
                reward+=80
                self.phase = 2
                return self._get_obs(),reward,True,False,info

        return self._get_obs(), reward, False, truncated, info
        """
        return obs,reward,terminated,truncated,info
        obs:姿态
        reward:奖励
        terminated:是否自然终止
        truncated:是否截断终止
        info:附加信息
        """

    def _get_obs(self):
        qpos = self.data.qpos[self.arm_qpos_adr].copy()[:]
        qval = self.data.qvel[self.arm_dof_adr].copy()[:]
        ee_pos = self.data.site_xpos[self.ee_id].copy()
        target_pos = np.array([self.box_pos[0],self.box_pos[1],self.box_pos[2]+0.25])
        rel_pos = target_pos - ee_pos
        rot_mat = self.data.geom_xmat[self.box_geom_id].reshape(3,3)
        box_edge_dir = rot_mat[:2,0].copy()
        palm_mat = self.data.geom_xmat[self.palm_id].reshape(3,3)
        gripper_y_axis = palm_mat[:2,1].copy()
        gripper_open_value = np.array([float(self.data.qpos[self.finger_qpos_adr])])
        #4+4+3+3+3+2+2+1 = 22

        return np.concatenate(
            [qpos ,qval ,ee_pos ,target_pos ,rel_pos ,box_edge_dir,gripper_y_axis,gripper_open_value]
        ).astype(np.float32)
        #数据必须为np.float32

    def _set_dist(self,dist):
        self.dist_threshold = dist

    def _set_angle(self,angle) :
        self.angle_threshold = angle