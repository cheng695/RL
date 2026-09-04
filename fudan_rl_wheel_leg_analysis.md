# fudan_rl_wheel_leg-main 方案阅读与 Isaac Lab 迁移建议

## 1. 项目整体方案

这个仓库是典型 `legged_gym + Isaac Gym Preview 4 + rsl_rl` 风格项目，分成两套任务：

- `plane/`：平地轮腿运动任务，核心目标是速度、yaw rate 和 base height 命令跟踪。
- `jump/`：跳跃任务，在相同轮腿基础上加入飞行、起跳伸腿、空中收腿、飞行高度等奖励。

核心代码位置：

- 平地任务配置：`fudan_rl_wheel_leg-main/plane/wheel_legged_gym/envs/base/legged_robot_config.py`
- 平地机器人特化配置：`fudan_rl_wheel_leg-main/plane/wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py`
- 平地环境实现：`fudan_rl_wheel_leg-main/plane/wheel_legged_gym/envs/base/legged_robot.py`
- 跳跃任务配置：`fudan_rl_wheel_leg-main/jump/wheel_legged_gym/envs/base/legged_robot_config.py`
- 跳跃环境实现：`fudan_rl_wheel_leg-main/jump/wheel_legged_gym/envs/base/legged_robot.py`

它的关键思路不是只看 root height，而是显式计算轮腿机构的虚拟腿长 `L0` 和虚拟腿角 `theta0`，然后用这些中间量做姿态/腿型约束。

## 2. 机器人、动作和命令

### 动作

`num_actions = 6`，对应：

- 左腿两个位置控制关节
- 左轮速度控制关节
- 右腿两个位置控制关节
- 右轮速度控制关节

动作在 `_compute_torques()` 中被拆成两类：

```python
pos_ref = actions * pos_action_scale
pos_ref[:, 2] *= 0
pos_ref[:, 5] *= 0

vel_ref = actions * vel_action_scale
vel_ref[:, :2] *= 0
vel_ref[:, 3:5] *= 0
```

也就是腿关节用 position target，轮子用 velocity target。最终还是统一走 PD/阻尼形式输出 torque。

### 命令

命令维度实际使用前三维：

```text
commands[:, 0] = lin_vel_x
commands[:, 1] = ang_vel_yaw
commands[:, 2] = height
```

平地任务配置：

```python
lin_vel_x = [-1.0, 1.0]
ang_vel_yaw = [-15, 15]
height = [0.10, 0.20]
resampling_time = 5.0
```

跳跃任务配置：

```python
lin_vel_x = [-2.1, 2.1]
ang_vel_yaw = [-2, 2]
height = [0.12, 0.15]
resampling_time = 20.0
```

命令是随机重采样，不是三角波连续扫动。

## 3. 虚拟腿几何

环境每步都会根据关节角计算两侧虚拟腿：

```python
theta1 = [dof_pos[:, 0], -dof_pos[:, 3]]
theta2 = [dof_pos[:, 1] + pi / 2, -dof_pos[:, 4] + pi / 2]

end_x = offset + l1 * cos(theta1) + l2 * cos(theta1 + theta2)
end_y = l1 * sin(theta1) + l2 * sin(theta1 + theta2)

L0 = sqrt(end_x^2 + end_y^2)
theta0 = atan2(end_y, end_x) - pi / 2
```

这说明它明确利用机构几何：

- `L0`：虚拟腿长
- `theta0`：虚拟腿相对竖直方向的摆角

这个点对你很重要：它不是“直接给腿长指令”，但 reward 里确实用到了腿型几何。也就是说，外部 command 是 base height/velocity/yaw，但内部 reward 可以用腿长和腿角做结构先验。

## 4. Observation

### Actor obs：25 维

`compute_proprioception_observations()` 拼接如下：

```text
base_ang_vel * ang_vel_scale            3
projected_gravity                       3
commands[:3] * commands_scale           3
dof_pos[[0,1,3,4]] - default_dof_pos    4
dof_vel                                 6
actions                                 6
------------------------------------------
total                                  25
```

注意：

- actor obs 没有直接给 `base_lin_vel`，这部分由历史编码器/critic 或隐式历史估计。
- 只给了 4 个腿关节的位置偏差，不给轮子位置偏差。
- 给了全部 6 个 dof velocity。
- 给了上一帧动作。
- `obs_history_length = 5`，策略类是 `ActorCriticSequence`，实际 actor 会使用 5 帧历史。

### Critic privileged obs

critic 额外得到：

- `base_lin_vel`
- actor obs
- 最近两帧 actions
- `dof_acc`
- terrain height scan
- torques
- base mass perturbation
- base COM
- randomized default dof offset
- friction
- restitution

这是典型 asymmetric actor-critic：actor 用可部署观测，critic 用仿真特权信息提高训练稳定性。

## 5. Plane Reward

平地任务启用的 reward scale：

```python
tracking_lin_vel = 1.0
tracking_lin_vel_enhance = 1.0
tracking_ang_vel = 1.0
tracking_ang_vel_enhance = 1.0
base_height = 1.0
nominal_state = -1.0
lin_vel_z = -1.0
ang_vel_xy = -0.20
orientation = -100.0
dof_vel = -5e-5
dof_acc = -2.5e-7
torques = -0.0001
action_rate = -0.01
action_smooth = -0.01
collision = -1.0
dof_pos_limits = -1.0
```

主要 reward 函数：

```python
tracking_lin_vel = exp(-(cmd_vx - base_vx)^2 / tracking_sigma)
tracking_lin_vel_enhance = exp(-(cmd_vx - base_vx)^2 / tracking_sigma / 10) - 1
tracking_ang_vel = exp(-(cmd_yaw - base_yaw_rate)^2 / tracking_sigma)
tracking_ang_vel_enhance = exp(-(cmd_yaw - base_yaw_rate)^2 / tracking_sigma / 10) - 1
base_height = exp(-(base_height - cmd_height)^2 / 0.001)
lin_vel_z = base_vz^2
ang_vel_xy = base_roll_pitch_rate^2
orientation = projected_gravity_xy^2
nominal_state = (theta0_left - theta0_right)^2
```

几个值得注意的点：

- `base_height` 用的是 `exp(-error^2 / 0.001)`，1/e 半宽约 3.16 cm，比你之前用的 `0.006` 更严格。
- `orientation = -100` 非常重，但单项 reward 又会被每步 clip 到 `clip_single_reward * dt`，所以实际不会无限大。
- `nominal_state` 不是固定关节角，而是约束左右虚拟腿角一致。
- 它有 `tracking_*_enhance = exp(... / 10) - 1`，这是负值增强项，用来拉开跟踪差距。

## 6. Jump Reward

跳跃任务的 reward 变化比较大，核心 scale：

```python
tracking_lin_vel = 1.0
tracking_lin_vel_enhance = 1.0
tracking_ang_vel = 1.0
flight = 0.15
encourage_jump = 1.0
base_height_flight = 6.0
leg_tuck = 1.7
takeoff_extend = 0.5
line_z = 6.0
pen_theta_no0 = -2.0
action_rate = -0.04
torques = -0.00005
orientation = -25.0
ang_vel_xy = -0.1
nominal_state = -1.0
collision = -1.0
```

跳跃 reward 的设计：

- 起跳阶段：`takeoff_extend` 奖励接触地面且向上速度足够时腿长接近 `0.31`。
- 空中阶段：`leg_tuck` 奖励空中腿长接近 `0.16`，也就是收腿。
- 飞行阶段：`flight` 直接奖励处于飞行状态。
- 飞行高度：`base_height_flight` 奖励 root z 接近 `0.65`。
- 向上速度：`line_z` 奖励飞行阶段正 z 速度。
- 腿角：`pen_theta_no0` 惩罚 `theta0` 偏离 0，让腿保持在身体下方。
- 对称性：`nominal_state = (theta0_left - theta0_right)^2 + 10 * (L0_left - L0_right)^2`。

所以 jump 方案更像阶段化 shaping：起跳伸腿、空中收腿、飞行高度、落地稳定。

## 7. Termination

平地任务 termination 主要是：

- 指定 body 接触力超过阈值并持续一段时间
- `projected_gravity[:, 2] > -0.1`，也就是机体明显翻倒
- episode timeout
- 地形边界 reset

它用 `fail_buf` 累计失败持续时间，而不是一触发就马上 reset：

```python
reset = fail_buf > fail_to_terminal_time_s / dt
```

这个对训练很友好：短暂接触/晃动不立即杀死 episode。

## 8. 对你当前任务的启发

你现在的问题是让机器人站高/蹲下。Fudan 方案给出的主要启发是：

1. 外部 command 仍然应该是 `base_height`，不用给腿长指令。
2. 但 reward 里可以使用机构几何，如 `L0`、`theta0` 或左右镜像约束。
3. base height reward 的 sigma 应该比较严格，Fudan 用 `0.001`，约 3.16 cm 半宽。
4. 不要只靠 `base_height` 正奖励，因为随机高度命令下“不动”也能拿部分分数。
5. 对轮腿机构，虚拟腿角/左右对称 reward 很有价值，比直接给固定关节角更自然。
6. Fudan 用随机高度命令，不是连续扫动；如果你要先学站高/蹲下，可以先试随机阶跃，再试慢速三角波。

## 9. 从 Gym 框架迁移到 Isaac Lab 框架

### 模块映射

| legged-gym 写法 | Isaac Lab manager 写法 |
|---|---|
| `LeggedRobotCfg.env` | `ManagerBasedRLEnvCfg` 的 scene/env 参数 |
| `_resample_commands()` | `CommandTerm`，例如 `UniformBaseHeightCommand` |
| `compute_proprioception_observations()` | `ObservationTermCfg` 多个 term |
| `_reward_xxx()` | `RewardTermCfg(func=...)` |
| `check_termination()` | `TerminationTermCfg(func=...)` |
| `_compute_torques()` | `ActionTermCfg`，自定义 action 或组合 position/velocity action |
| `_post_physics_step_callback()` | metrics、command update、sensor callback，尽量拆进 observation/reward/command term |
| domain randomization | `EventTermCfg(mode=startup/reset/interval)` |

### 迁移 obs

建议在 Isaac Lab 里复现 actor obs：

```text
base_ang_vel                 3
projected_gravity            3
base_velocity command        2 或 3
base_height command          1
leg joint pos rel            4
all joint vel                6
last action                  6
```

如果你现在的 `base_height` command 是 `[height, rate]`，actor obs 可以保留 `[height, rate]`，这比 Fudan 更适合连续扫动。

critic 可以加：

```text
base_lin_vel
height scan
torques
dof_acc
contact state
domain randomization params
```

### 迁移 reward

建议先迁移平地任务的核心项，而不是 jump 项：

```text
base_height_tracking
base_height_velocity_error
theta0_equ_0 或 mirrored_leg_pattern
left/right L0 symmetry
orientation
ang_vel_xy
action_rate
joint_vel
collision/contact
```

对于你当前的两轮闭链轮腿，更推荐把 Fudan 的 `L0/theta0` 思路换成你已经在 Lab 里实现的几何函数：

- `base_height_commanded_leg_vertical_l2`
- `mirrored_leg_joint_pattern_l2`
- `wheel_x_separation_l2`

如果可以从 CAD/URDF 明确两连杆长度和主动关节定义，最好进一步实现 Fudan 风格的 `L0/theta0`，因为它比从 body pose 估计更稳定、噪声更小。

### 迁移动作

Fudan 的动作是混合控制：

- 腿：position target
- 轮：velocity target

Isaac Lab 里最好继续保持这个结构：

- 一个 `JointPositionActionCfg` 或自定义 bounded position action 控制腿关节
- 一个 `JointVelocityActionCfg` 控制轮关节

不要把轮子也做 position control。

### 迁移训练参数

Fudan 使用：

```python
obs_history_length = 5
policy_class_name = "ActorCriticSequence"
init_noise_std = 0.5
entropy_coef = 0.01
num_steps_per_env = 48
learning_rate = 1e-3
gamma = 0.99
desired_kl = 0.005
```

你现在如果 action std 很快掉到 `0.02`，可以参考它提高探索：

```python
init_noise_std = 0.5
entropy_coef = 0.005 ~ 0.01
num_steps_per_env = 48
```

### 推荐迁移顺序

1. 先复现 actor obs：base ang vel、gravity、command、joint pos/vel、last action、history。
2. 再复现 command：先用随机高度命令 `[0.27, 0.35]`，不要一开始用连续三角波。
3. 迁移基础 reward：base height、orientation、ang vel xy、action rate、joint vel、contact。
4. 加机构几何 reward：左右镜像、竖直支撑高度、左右腿高度一致。
5. 训练稳定后再恢复速度/yaw command。
6. 最后再考虑跳跃任务里的阶段化 reward。

## 10. 对你当前 Lab 代码的建议

短期建议：

- 把 `base_height.weight = 8.0` 改回 `5.0`，否则“不动也拿分”的部分也被放大。
- 保留 `base_height_vel.weight = -2.0`，让它必须产生 z 速度。
- `BASE_HEIGHT_TRACKING_SIGMA` 可试 `0.001`，贴近 Fudan。
- `entropy_coef` 可提高到 `0.005` 或 `0.01`，参考 Fudan，防止 action std 过早塌缩。
- 如果要参考 Fudan，高度命令先改成随机阶跃而不是三角波：每 3-5 秒随机采样一个目标高度。

更结构化的方向：

- 用 base height command 做外部任务目标。
- 用虚拟腿几何做内部 shaping。
- 不直接奖励固定关节角。
- 奖励左右镜像、同侧同步、腿竖直支撑高度接近目标。
- 训练初期减少水平速度/yaw 等额外目标，让它先学会高度可控。
