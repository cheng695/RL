# RL Training Dashboard

使用 `env_isaaclab` 环境启动：

```bash
cd /home/whc/桌面/RL
/home/whc/miniconda3/envs/env_isaaclab/bin/python \
  rl-training-handoff/dashboard/server.py \
  --log-root /home/whc/桌面/RL/logs/rsl_rl
```

打开 <http://127.0.0.1:8765>。页面会自动扫描 `logs/rsl_rl/*/*` 下的 TensorBoard event 文件和 checkpoint。

在线 Agent 播放请直接使用 Isaac Sim 窗口和 `Wheel-Leg/scripts/rsl_rl/play.py`；本目录的网页仅用于查看训练日志和 checkpoint。
