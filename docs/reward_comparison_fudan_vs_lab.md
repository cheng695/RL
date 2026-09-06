# Wheel-Leg Reward 对比：Fudan Gym vs 当前 Isaac Lab

本文对比两个实现：

- Fudan: `fudan_rl_wheel_leg-main/plane/wheel_legged_gym/envs/base/legged_robot.py`
- 当前: `Wheel-Leg/envs/wheel_leg/mdp/rewards.py` 与 `Wheel-Leg/envs/wheel_leg/wheel_leg_env_cfg.py`

记号：

- `v_x`: base 坐标系前向速度
- `v_z`: base 坐标系竖直速度
- `w_z`: base 坐标系 yaw rate
- `w_xy`: roll/pitch 角速度
- `g_xy`: projected gravity 的 x/y 分量
- `cmd_vx`: 速度命令 x
- `cmd_yaw`: yaw rate 命令
- `cmd_h`: base height 命令
- `a_t`: 当前 action

## 总体差别

当前 reward 已经不是 Fudan 的完全等价迁移，主要差别有：

1. 当前多了高度命令到腿部点几何的先验 `leg_vertical_command`。
2. 当前多了左右腿/轮位置防作弊项：`mirrored_leg_pattern`、`wheel_x_separation`、`wheel_air`。
3. 当前多了 `no_yaw_spin`：只在 `cmd_yaw≈0` 时抑制自转。
4. 当前把二阶 action smooth 拆成腿和轮两个项，而且权重比 Fudan 大。
5. 当前没有启用 Fudan 的 `tracking_lin_vel_enhance`、`tracking_ang_vel_enhance`、`dof_acc`、`dof_pos_limits`、`collision` 完全同款项。
6. Fudan 会在 `_prepare_reward_function()` 中把所有 reward scale 乘以 `dt`，并对单项 reward 做 clip；当前 Isaac Lab 配置里权重没有显式乘 `dt`，日志数值不能直接和 Fudan 的 raw scale 一比一对应。

## Fudan Plane Reward

Fudan 配置中的 active scales：

| 项 | 权重 | 含义 | 公式 |
|---|---:|---|---|
| `tracking_lin_vel` | `1.0` | 跟踪前向速度 `cmd_vx` | `exp(-(cmd_vx - v_x)^2 / sigma)` |
| `tracking_lin_vel_enhance` | `1.0` | 更宽松的前向速度辅助项 | `exp(-(cmd_vx - v_x)^2 / (10*sigma)) - 1` |
| `tracking_ang_vel` | `1.0` | 跟踪 yaw rate | `exp(-(cmd_yaw - w_z)^2 / sigma)` |
| `tracking_ang_vel_enhance` | `1.0` | 更宽松的 yaw 辅助项 | `exp(-(cmd_yaw - w_z)^2 / (10*sigma)) - 1` |
| `base_height` | `1.0` | 跟踪 base 高度命令 | `exp(-(base_h - cmd_h)^2 / 0.001)` |
| `nominal_state` | `-1.0` | 左右虚拟腿摆角一致 | `(theta0_l - theta0_r)^2` |
| `lin_vel_z` | `-1.0` | 抑制上下跳 | `v_z^2` |
| `ang_vel_xy` | `-0.20` | 抑制 roll/pitch 角速度 | `sum(w_xy^2)` |
| `orientation` | `-100.0` | 抑制机身倾斜 | `sum(g_xy^2)` |
| `dof_vel` | `-5e-5` | 抑制腿关节速度 | `sum(dof_vel_leg^2)` |
| `dof_acc` | `-2.5e-7` | 抑制关节加速度 | `sum(dof_acc^2)` |
| `torques` | `-1e-4` | 抑制力矩 | `sum(tau^2)` |
| `action_rate` | `-0.01` | 一阶 action 平滑 | `sum((a_t - a_{t-1})^2)` |
| `action_smooth` | `-0.01` | 二阶 action 平滑，只作用腿部 action | `sum((a_t - 2a_{t-1} + a_{t-2})^2)` |
| `collision` | `-1.0` | 惩罚非期望 body 接触 | `count(contact_force > 0.1)` |
| `dof_pos_limits` | `-1.0` | 惩罚腿关节接近限位 | 超出 soft limit 的距离和 |

Fudan 的 `sigma = 0.25`。注意 Fudan 命令数组是：

```text
commands[:, 0] = vx
commands[:, 1] = yaw rate
commands[:, 2] = base height
```

## 当前 Isaac Lab Reward

当前配置中的 active reward：

| 项 | 权重 | 含义 | 公式 |
|---|---:|---|---|
| `tracking_lin_vel` | `1.0` | 跟踪前向速度 `cmd_vx` | `exp(-(cmd_vx - v_x)^2 / LIN_VEL_TRACKING_SIGMA)` |
| `tracking_ang_vel` | `1.0` | 跟踪 yaw rate | `exp(-(cmd_yaw - w_z)^2 / YAW_RATE_TRACKING_SIGMA)` |
| `no_yaw_spin` | `-1.0` | `cmd_yaw≈0` 时抑制自转，不锁 yaw 角度 | `I(|cmd_yaw|<=0.05) * max(|w_z|-0.08,0)^2` |
| `base_height` | `1.0` | 跟踪随机 base height 命令 | `exp(-(base_z - cmd_h)^2 / BASE_HEIGHT_TRACKING_SIGMA)` |
| `leg_vertical_command` | `-0.2` | 用点几何约束腿部竖直支撑高度跟随 `cmd_h` | `sum(((vertical_support - target_vertical)_deadband / 0.02)^2)` |
| `nominal_state` | `-1.0` | Fudan nominal_state 的点几何版本，左右虚拟腿摆角一致 | `(theta_l - theta_r)^2` |
| `mirrored_leg_pattern` | `-0.05` | 约束腿部关节满足左侧同号、右侧同号、左右镜像 | `(LF-LR)^2 + (RF-RR)^2 + (LF+RF)^2 + (LR+RR)^2` |
| `lin_vel_z` | `-1.0` | 抑制上下跳 | `v_z^2` |
| `ang_vel_xy` | `-0.2` | 抑制 roll/pitch 角速度 | `sum(w_xy^2)` |
| `orientation` | `-20.0` | 抑制机身倾斜 | `sum(g_xy^2)` |
| `base_contact` | `-10.0` | base link 碰地惩罚 | `1 if contact_force > threshold else 0` |
| `wheel_air` | `-1.0` | 任意轮子离地惩罚 | `1 if any wheel not in contact else 0` |
| `wheel_x_separation` | `-1.0` | 防止左右轮在 base x 方向劈叉 | `max(|x_L - x_R|-deadband,0)^2` |
| `leg_joint_vel` | `-5e-5` | 抑制腿部关节速度 | `sum(qdot_leg^2)` |
| `joint_torques` | `-1e-4` | 抑制 4 腿 + 2 轮执行器力矩 | `sum(tau_controlled^2)` |
| `action_rate` | `-0.01` | 一阶 action 平滑，作用全部 6 维 action | `sum((a_t - a_{t-1})^2)` |
| `leg_action_second_order` | `-0.05` | 腿部 action 二阶平滑 | `sum((a_leg,t - 2a_leg,t-1 + a_leg,t-2)^2)` |
| `wheel_action_second_order` | `-0.05` | 轮毂 action 二阶平滑 | `sum((a_wheel,t - 2a_wheel,t-1 + a_wheel,t-2)^2)` |

当前使用的关键参数：

```python
LIN_VEL_TRACKING_SIGMA = 0.25
YAW_RATE_TRACKING_SIGMA = 1.0
BASE_HEIGHT_TRACKING_SIGMA = 0.0016
NO_YAW_COMMAND_RATE_DEADBAND = 0.05
NO_YAW_ACTUAL_RATE_DEADBAND = 0.08
```

当前命令数组是：

```text
base_velocity[:, 0] = vx
base_velocity[:, 1] = vy
base_velocity[:, 2] = yaw rate
base_height[:, 0] = base height
```

## 逐项差异

### 1. 速度跟踪

Fudan:

```python
tracking_lin_vel = exp(-(cmd_vx - v_x)^2 / 0.25)
tracking_lin_vel_enhance = exp(-(cmd_vx - v_x)^2 / 2.5) - 1
```

当前:

```python
tracking_lin_vel = exp(-(cmd_vx - v_x)^2 / 0.25)
```

差异：

- 当前没有 `tracking_lin_vel_enhance`。
- 当前练 vx 时把 yaw command 设为 0，且 command sampler 可以做 vx/yaw 互斥。
- 如果 vx 学得慢，可以考虑恢复 Fudan 的 enhance 项，但要注意它是负到 0 的辅助项。

### 2. Yaw 跟踪

Fudan:

```python
tracking_ang_vel = exp(-(cmd_yaw - w_z)^2 / 0.25)
tracking_ang_vel_enhance = exp(-(cmd_yaw - w_z)^2 / 2.5) - 1
```

当前:

```python
tracking_ang_vel = exp(-(cmd_yaw - w_z)^2 / 1.0)
no_yaw_spin = I(|cmd_yaw|<=0.05) * max(|w_z|-0.08,0)^2
```

差异：

- 当前 yaw tracking sigma 更宽，目的是避免早期 yaw 误差大时 reward 接近 0。
- 当前额外加了 `no_yaw_spin`，用于 vx-only 阶段抑制自转。
- 当前没有 `tracking_ang_vel_enhance`。

### 3. 高度跟踪

Fudan:

```python
base_height = exp(-(base_h - cmd_h)^2 / 0.001)
```

当前:

```python
base_height = exp(-(base_z - cmd_h)^2 / 0.0016)
leg_vertical_command = -0.2 * sum(((vertical_support - target_vertical)_deadband / 0.02)^2)
```

差异：

- 当前 base height sigma 更宽一点：`0.0016` 对应约 4 cm 到 `1/e`。
- 当前额外把 `cmd_h` 映射到 hip point -> wheel center 的竖直支撑高度，避免只奖励 root z 时腿不主动伸缩。
- Fudan 没有这个点几何腿竖直高度项。

### 4. 姿态和稳定

Fudan:

```python
lin_vel_z = v_z^2
ang_vel_xy = sum(w_xy^2)
orientation = sum(g_xy^2), weight=-100
```

当前:

```python
lin_vel_z = v_z^2
ang_vel_xy = sum(w_xy^2)
orientation = sum(g_xy^2), weight=-20
```

差异：

- 当前 orientation 权重比 Fudan 小很多：`-20` vs `-100`。
- 这是因为当前 USD/闭链模型早期如果姿态惩罚过大，可能压制高度和腿部探索。

### 5. 腿型/虚拟腿

Fudan:

```python
nominal_state = (theta0_l - theta0_r)^2
```

当前:

```python
nominal_state = (theta_l - theta_r)^2
mirrored_leg_pattern = (LF-LR)^2 + (RF-RR)^2 + (LF+RF)^2 + (LR+RR)^2
wheel_x_separation = max(|x_L-x_R|-deadband,0)^2
```

差异：

- Fudan 使用其模型内部虚拟腿角 `theta0`。
- 当前两连杆虚拟腿拟合闭链 USD 误差较大，因此改用 hip point -> wheel center 点几何。
- 当前还加了关节镜像和左右轮 x 分离惩罚，主要是为了防止闭链结构出现劈叉/爆开策略。

### 6. 平滑和能耗

Fudan:

```python
dof_vel = sum(qdot_leg^2)
dof_acc = sum(qddot^2)
torques = sum(tau^2)
action_rate = sum((a_t-a_t-1)^2)
action_smooth = leg action 二阶差分
```

当前:

```python
leg_joint_vel = sum(qdot_leg^2)
joint_torques = sum(tau_controlled^2)
action_rate = sum((a_t-a_t-1)^2)
leg_action_second_order = 腿 action 二阶差分
wheel_action_second_order = 轮 action 二阶差分
```

差异：

- 当前没有启用 `dof_acc`。
- 当前二阶 action smooth 分成腿和轮，但当前权重都是 `-0.05`，比 Fudan 的 `-0.01` 更强。
- 如果 vx/yaw 响应变慢，优先降低 `wheel_action_second_order`，例如 `-0.05 -> -0.01` 或 `-0.005`。

### 7. 接触项

Fudan:

```python
collision = count(non-foot contact force > 0.1)
feet_contact_forces = sum(max(contact_force - max_contact_force, 0))
```

当前:

```python
base_contact = base_link 碰地
wheel_air = 任意轮子离地
```

差异：

- 当前更针对轮腿平衡：base 不能碰地，轮子最好保持接触。
- Fudan 更通用地惩罚非期望 body collision 和过大接触力。

## 当前最需要注意的点

### `no_yaw_spin` 与 `tracking_ang_vel`

当前 `COMMAND_ANG_VEL_Z_RANGE = (0.0, 0.0)` 时，`tracking_ang_vel` 本身已经在奖励 `w_z -> 0`。但指数 reward 在误差较大时会趋近 0，梯度弱；`no_yaw_spin` 是 L2 惩罚，能更直接压住 vx-only 阶段自转。

这不是锁 yaw 角度，只是压 yaw rate。未来打开 yaw command 后：

```python
COMMAND_ANG_VEL_Z_RANGE = (-1.5, 1.5)
```

只要 `abs(cmd_yaw) > 0.05`，`no_yaw_spin` 自动关闭。

### 当前二阶 smooth 可能偏强

当前：

```python
leg_action_second_order = -0.05
wheel_action_second_order = -0.05
```

Fudan：

```python
action_smooth = -0.01
```

如果观察到：

- vx 跟踪变慢
- yaw 起不来
- 轮速响应很迟钝

建议先把轮毂二阶 smooth 降到：

```python
wheel_action_second_order = -0.005 或 -0.01
```

腿部如果还抖，再单独调腿部项。

## 如果要更接近 Fudan

可以考虑：

1. 恢复 `tracking_lin_vel_enhance` 和 `tracking_ang_vel_enhance`。
2. 将 `orientation` 从 `-20` 逐步提高到 `-50`，稳定后再试 `-100`。
3. 增加 `dof_acc`，但权重要很小，如 `-2.5e-7`。
4. 将二阶 smooth 权重调回 Fudan 附近，尤其轮子不要太大。
5. 保留当前点几何高度项，因为这是针对当前闭链 USD 的必要补充。

