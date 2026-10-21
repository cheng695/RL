# 速度跟踪续训与轮子抖动诊断

旋转跟踪的排查、参数调整及固定命令对照结果见 [旋转速度跟踪问题与解决过程](2026-10-21-旋转速度跟踪问题与解决过程.md)。

左右腿质量对称实验、模型切换及质心随机化对照开关见 [uz05_mass_symmetry.md](uz05_mass_symmetry.md)。
训练默认模型为 equalized；PLAY 默认根据 checkpoint 配置恢复质量版本与质心随机化设置。
显式设置 `UZ05_MASS_VARIANT=original` 可强制使用原始模型。

当前配置：站立 20%、直线 40%、原地转向 20%、行进转向 20%。
前后方向各半；带平移的命令幅值按低速/中速/高速 60/25/15 采样，
边界分别为 0.08、0.5、1.0、2.5 m/s。
非零 yaw 命令幅值按 60/25/10/5 分段采样，边界为 0.15、1.0、2.5、5.0、12.0 rad/s，左右转各半。
这是质量实验从头训练时的低速优先设置，比例不会随训练进度自动改变。
此比例作用于原地转向和行进转向，站立与直线的 yaw 命令仍为零。
高度继续在 0.25～0.40 m 每回合采样一次。
以上为轮速投影前的采样比例；混合运动超出轮速可行域后会等比例缩小。
额外的 `straight_pitch_penalty` 权重暂设为 0；通用姿态奖励继续生效。
用户随后从头训练，反馈 vx 跟踪明显改善；新策略的同工况 CSV 定量对比及旋转表现仍待确认。
后续 yaw=+0.5 rad/s 固定命令记录的实际均值为 0.098 rad/s、MAE 约 0.402 rad/s；
因此将转向样本占比从 20% 提高到 40%，并增加低角速度样本。
这是待训练验证的调整，不能认为已解决 yaw 跟踪不足。
问题、证据、修改与效果详见 [轮速振荡与速度跟踪排查总结](2026-10-20-轮速振荡与速度跟踪排查总结.md)。

### 轮速目标振荡修正（2026-10-20）

- `wheel_joint_vel.max_velocity_rate` 从 400 降至 100 rad/s²；
  在 0.02 s 控制周期下，单步目标变化上限为 2 rad/s。
- 新增 `wheel_target_smooth`：权重 −0.1，对左右轮最终目标变化
  `mean((Δω_cmd / 2.0)²)` 施加惩罚。这里的 2.0 单位是 rad/s，
  随控制周期修改时需重新考虑尺度；两轮都达到单步变化上限时，
  加权值为 −0.1（奖励管理器乘时间步长之前）。
- 由动作项保存最终目标差分，reset 时按环境清零；
  恒定非零轮速目标不会持续产生这一惩罚。
- 原有原始动作二阶差分惩罚保留；观测维度不变，可直接续训原 checkpoint。
- 训练和 play 都采用新的限制。即使播放旧 checkpoint，控制条件也已经改变；
  原先 `before_forward_02.csv` 是旧限制下的基线，应保留，并为新记录使用不同文件名。
- 较严的限制可能影响平衡恢复，需同时对比速度误差、抖动和失败次数。

## 续训

以下从已学会 vx 的策略续训。运行中的进程不会自动加载新配置。

```bash
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Flat-v0 \
  --num_envs 1024 \
  --max_iterations 1500 \
  --run_name yaw_tracking_continue \
  --resume \
  --checkpoint "/home/whc/桌面/RL/logs/rsl_rl/uz05_flat/2026-10-20_18-45-34_flat_baseline/model_2900.pt" \
  --headless
```

## 固定命令记录

PLAY 命令范围：vx 为 −2.5～2.5 m/s，vy 为 0，yaw 为 −12～12 rad/s，
高度为 0.25～0.40 m。键盘速度档位上限读取 PLAY 配置，I/K 调节高度也读取这一范围。
平移与转向同时下发时仍经过左右轮转速可行域投影，HUD 的 effective 命令可能小于 target。
命令投影使用轮速上限 50 rad/s，与当前动作目标幅值一致，即
`abs(vx) + 0.210335 * abs(yaw) <= 2.75`。
纯直线 ±2.5 m/s 和原地转向 ±12 rad/s 均可保留；超出该边界的组合命令等比例缩小。

分别对修改前后的 checkpoint 执行同一组命令。CSV 路径必须不存在，避免覆盖记录。

```bash
python Wheel-Leg/scripts/rsl_rl/play.py \
  --task MyRobot-Velocity-Flat-Play-v0 \
  --num_envs 1 \
  --checkpoint "/path/to/model.pt" \
  --seed 42 \
  --command_vx 0.2 \
  --command_yaw 0 \
  --command_height 0.30 \
  --no_keyboard --no_hud \
  --motion_trace_csv logs/motion_traces/before_forward_02.csv \
  --motion_trace_steps 1000 \
  --headless
```

其他工况使用不同文件名，并替换命令：

| 工况 | command_vx | command_yaw |
|---|---:|---:|
| 后退低速 | -0.2 | 0 |
| 前进中速 | 0.5 | 0 |
| 后退中速 | -0.5 | 0 |
| 左右原地转向 | 0 | +0.5 / -0.5 |

## 数据含义

- 每个控制步记录 env0 的四个腿关节、两个轮子，按关节名字区分列。
- `policy_raw`：裁剪前的网络动作；`action_clipped`：动作项收到的动作。
- `q_target_rad` / `omega_target_rad_s`：最终下发的位置/速度目标，已经过变化率限制。
  腿部主要看位置目标，轮子主要看速度目标。
- 记录实际关节角度、轮速、机体速度、yaw 角速度、roll/pitch、高度。
- `tau_estimate_nm` 使用 `robot.data.applied_torque`。当前为隐式执行器，
  该字段是执行器估计值，不能当作真实电机传感器或 PhysX 求解力矩。
- `estimated_effort_at_limit` 表示估计力矩达到配置限值的 99%，不是策略动作限幅率。
- 每步记录前一动作执行后的状态及对应命令。自动 reset 帧标记
  `done=1,state_valid=0`，物理状态留空，分析时排除这些帧。
- 同名 JSON 记录 checkpoint、时间步长、关节顺序及采样说明。
- 采样频率是控制频率，不能排除控制步之间的物理子步振荡。

目标轮速自身振荡时，先检查策略和奖励；目标平稳但实际轮速振荡时，再检查驱动和接触。
比较前后策略时应分别统计稳定阶段的速度误差、轮速变化、roll/pitch，
并单独比较失败次数，避免只保留成功片段。
