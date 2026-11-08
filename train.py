import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.arm_env import ArmEnv,CurriculumCallback

N_ENVS = 16
TOTAL_STEPS = 1000_000

if __name__ == "__main__":
    env = make_vec_env(ArmEnv, n_envs=N_ENVS, seed=0)

    model = PPO.load("D:/programming/python/mujoco_xml/scene/saves/phase1.zip",env=env)
    '''
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        device="cpu",
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=0,
    )'''
    
    

    print("开始训练机械臂到达 + 指向任务...")
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[CurriculumCallback()],
        progress_bar=False,
    )
    model.save("ppo_arm_reach")
    print("训练完成，模型已保存为 ppo_arm_reach.zip")
