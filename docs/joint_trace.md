# 关节目标与实际状态记录

Play 时可以选一个关节，逐控制步记录目标角度、实际角度、实际角速度和 PD 力矩估计，并生成曲线。

## 记录

在仓库根目录激活 `env_isaaclab` 后运行：

```bash
python Wheel-Leg/scripts/rsl_rl/play.py \
  --task MyRobot-Velocity-Flat-Play-v0 \
  --num_envs 1 \
  --checkpoint "/path/to/model.pt" \
  --record_joint left_J_chassis_link2 \
  --joint_trace_csv logs/joint_traces/left_leg_trace.csv
```

将 `--record_joint` 替换为想观察的关节。腿关节名可从 `my_robot_.py` 中的 `LEG_JOINTS` 查看。运行时仍可使用 Play 的键盘控制；每个控制步都会写一行 CSV。若省略 `--joint_trace_csv`，文件会自动写入 `logs/joint_traces/`。

CSV 字段包括：

- `q_target_rad`：关节目标角度，单位 rad。
- `q_actual_rad`：仿真读到的实际关节角度，单位 rad。
- `qdot_actual_rad_s`：实际关节角速度，单位 rad/s。
- `tau_pd_estimate_nm`：用配置的刚度和阻尼计算并按 effort limit 裁剪的 PD 力矩估计，单位 N·m。它不是 PhysX 实测电机力矩。
- `done`：该控制步是否触发 episode 终止/reset。

## 绘图

Play 退出并写完 CSV 后运行：

```bash
python Wheel-Leg/scripts/plot_joint_trace.py logs/joint_traces/left_leg_trace.csv
```

脚本在 CSV 同目录生成同名 PNG，包含目标/实际角度、实际角速度和 PD 力矩估计三张曲线。也可以用 `--output /path/to/plot.png` 指定图片路径。
