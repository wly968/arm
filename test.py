import mujoco
import mujoco.viewer
import numpy as np
import time
from stable_baselines3 import PPO
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.arm_env import ArmEnv

def main():
    print("正在加载模型和物理环境...")
    
    # 1. 实例化环境
    env = ArmEnv()
    
    # 2. 加载训练好的 PPO 模型
    # 注意：请确保 ppo_arm_reach.zip 和 test.py 在同一个目录下
    try:
        model = PPO.load("ppo_arm_reach", env=env)
    except FileNotFoundError:
        print("❌ 未找到模型文件 ppo_arm_reach.zip，请确认是否训练完成并保存！")
        return

    print("✅ 模型加载成功！启动可视化窗口...")
    
    # 3. 启动 MuJoCo 交互式查看器
    with mujoco.viewer.launch_passive(env.model, env.data,show_left_ui=False, show_right_ui=False) as viewer:
        # 隐藏两侧的 UI 面板，让画面更纯净
        viewer.show_left_ui = False
        viewer.show_right_ui = False
        
        # 调整初始视角，方便观察整个传送带和机械臂
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 90
        
        # 初始化环境
        obs, _ = env.reset()
        episode_reward = 0.0
        step_count = 0
        
        while viewer.is_running():
            # 4. 让 AI 做出决策 (deterministic=True 关闭随机探索)
            action, _states = model.predict(obs, deterministic=True)
            
            # 5. 在环境中执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # 6. 刷新画面
            viewer.sync()
            time.sleep(1/60) # 控制播放速度
            
            # 7. 判断本局是否结束
            if terminated or truncated:
                status = "🎯 成功到达目标上方" if terminated else "⏱️ 超时未到达"
                print(f"{status} | 步数: {step_count}, 累计奖励: {episode_reward:.2f}")
                
                # 如果是因为撞飞或 NaN 报错终止
                if info.get("nan"):
                    print("💥 发生了物理爆炸 (NaN)")
                
                time.sleep(0.5) # 稍微停顿一下看结果
                
                # 重置环境，开始下一局
                obs, _ = env.reset()
                episode_reward = 0.0
                step_count = 0

if __name__ == "__main__":
    main()