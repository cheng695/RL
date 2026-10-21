# Wheel-Leg 平地速度任务

当前项目仅保留平地速度训练与交互运行。场景使用平面碰撞地面，不启用高度扫描、生成式地形或地形课程。

机器人配置直接引用 `UZ05_MJCF_real_params/xmls/uz05_isaac.xml`。启动训练或 Play 时，Isaac Lab 会通过 `MjcfFileCfg` 自动转换为 `Wheel-Leg/assets/uz05/uz05.usd`，并启用实例化。项目转换适配器会先处理 MJCF 闭链约束，再调用 Isaac Lab 转换器。

## 训练

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Flat-v0 \
  --num_envs 1024 \
  --max_iterations 3000 \
  --run_name flat_baseline \
  --headless
```

固定速度/高度的自动评估及统计修正说明见 [tracking_evaluation.md](tracking_evaluation.md)。

当前高度/速度跟踪问题的原因、改动及分阶段训练命令见 [高度速度跟踪问题与解决方案](2026-10-18-高度速度跟踪问题与解决方案.md)。

## Play

将 checkpoint 路径替换为训练输出中的模型文件：

```bash
python Wheel-Leg/scripts/rsl_rl/play.py \
  --task MyRobot-Velocity-Flat-Play-v0 \
  --num_envs 1 \
  --checkpoint "/path/to/model.pt"
```

Play 默认只在机器人倒地时自动重置；可用键盘控制速度指令。

直行专项续训、左右轮/四腿诊断记录及固定命令对照方法见 [motion_trace.md](motion_trace.md)。

当前默认使用左右腿质量对称实验版。设置 `UZ05_MASS_VARIANT=original` 可恢复原始质量；
质量对照时，两版均设置 `UZ05_DISABLE_BASE_COM=1` 以关闭机身质心随机化。
实验方法与命令见 [uz05_mass_symmetry.md](uz05_mass_symmetry.md)。

平地直线运动
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Flat-v0 \
  --num_envs 1024 \
  --max_iterations 3000 \
  --run_name low_speed_continue \
  --resume \
  --checkpoint "/home/whc/桌面/RL/logs/rsl_rl/uz05_flat/2026-10-20_12-23-57_height_pitch_split_tune/model_2999.pt" \
  --headless

平地运动
/home/whc/桌面/RL/logs/rsl_rl/uz05_flat/2026-10-20_22-50-36_yaw_tracking_continue/model_5899.pt

复杂地形续训、姿态/高度处理和独立评估见 [轻度复杂地形训练](gentle_terrain_training.md)。
