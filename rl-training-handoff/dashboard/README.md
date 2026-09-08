# RL Training Dashboard

使用 `env_isaaclab` 环境启动：

```bash
cd /home/whc/桌面/RL
/home/whc/miniconda3/envs/env_isaaclab/bin/python \
  rl-training-handoff/dashboard/server.py \
  --log-root /home/whc/桌面/RL/logs/rsl_rl
```

打开 <http://127.0.0.1:8765>。页面会自动扫描 `logs/rsl_rl/*/*` 下的 TensorBoard event 文件和 checkpoint。

## 在线 Agent 预览

在线预览需要启动 Isaac Lab 的 `play.py`。它会加载 checkpoint，同时启动一个浏览器控制桥：

```bash
cd /home/whc/桌面/RL/Wheel-Leg

/home/whc/miniconda3/envs/env_isaaclab/bin/python \
  scripts/rsl_rl/play.py \
  --task MyRobot-Velocity-Flat-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/wheel_leg/<run>/model_1050.pt \
  --web_port 8766 \
  --real_time
```

然后打开 <http://127.0.0.1:8766>。Isaac Lab 窗口仍然是物理真值，网页用于拖动目标速度/高度和观察实时反馈。
