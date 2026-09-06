# wheeled-legged_RL-main 与当前 Wheel-Leg 任务对比

本文对比两个项目的平地轮腿训练任务：

- 开源项目：`/home/whc/桌面/RL/wheeled-legged_RL-main`
- 当前项目：`/home/whc/桌面/RL/Wheel-Leg`

重点只看三件事：`reward`、`observation`、`reset`。开源项目代码里有很多 rough terrain、state machine、standup、NP3O、play/debug 分支，本文以 `wheelbipe_V14` 平地主任务为主。

## 参考文件

开源项目主要读取：

- `wheeled-legged_RL-main/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/env_cfg.py`
- `wheeled-legged_RL-main/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/cfg_utils.py`
- `wheeled-legged_RL-main/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py`
- `wheeled-legged_RL-main/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env_cfg.py`

当前项目主要读取：

- `Wheel-Leg/envs/wheel_leg/wheel_leg_env_cfg.py`
- `Wheel-Leg/envs/wheel_leg/mdp/rewards.py`
- `Wheel-Leg/envs/wheel_leg/mdp/observations.py`

## 总体结论

`wheeled-legged_RL-main` 是 DirectRLEnv 风格：obs、reward、reset 都集中写在 env 里，reward 先计算一大批 `rew_*`，再用 `cfg.rewards` 的 OrderedDict 选择启用项和权重。

当前 `Wheel-Leg` 是 Isaac Lab ManagerBasedRLEnv 风格：obs、reward、reset 分别由 `ObsTerm`、`RewTerm`、`EventTerm` 配置，结构更清晰，也更容易做最小修改。

两者最大的差别不是 reward 名字，而是训练假设：

- 开源 V14 更偏“速度/姿态/高度联合任务”，有强速度跟踪、强姿态门控、较复杂 reset 和域随机化。
- 当前 `Wheel-Leg` 已经加入了很多针对我们问题的补丁项，例如 `stationary_vel`、`com_alignment`、`root_position`、`straight_yaw_rate`、`straight_wheel_diff`、`leg_vertical`。
- 开源 V14 的 actor policy obs 默认不直接放 `base_lin_vel_b` 和实际高度 `obs_height`，这些主要在 critic/privileged obs 里；当前 `Wheel-Leg` 的 actor obs 已经包含 `base_lin_vel`，速度跟踪信息更直接。

## Observation 对比

### 开源项目 V14

V14 基础配置里定义：

- `V14_BASE_POLICY_OBS_DIM = 28`
- `V14_BASE_PRIVILEGED_OBS_DIM = 32`

policy obs 的主要组成来自 `_get_observations()`：

| 项 | 含义 |
|---|---|
| `command` | 速度命令，通常为 `[vx_cmd, vy_cmd, yaw_cmd]` |
| `task_flag` | 可选任务标志，默认 V14 basic dim 中通常为 0 维 |
| `height_cmd` | base height 期望，高度命令会乘 `5.0` scale |
| `root_ang_vel_b` | base body-frame 角速度 |
| `projected_gravity_b` | 重力方向投影到 base frame，用来感知 roll/pitch |
| `joint_pos` | 受控关节相对默认角度，可能经过 joint pos encoding |
| `joint_vel_leg` | 腿部关节速度，scale 约 `0.1` |
| `joint_vel_wheel` | 轮毂关节速度，scale 约 `0.1` |
| `actions` | 当前/上一帧 policy action |

critic/privileged obs 会额外加入：

| 项 | 含义 |
|---|---|
| `root_lin_vel_b` | base body-frame 线速度 |
| `obs_height` | 实际观测高度 |
| `spring_force` | 弹簧力，启用 privileged extra 时使用 |
| `joint_torque` | 关节力矩 |
| `joint_acc` | 关节加速度 |
| `wheel_body_lin_vel` | 轮子 body 线速度 |
| `wheel_contact_force/state` | 轮子接触力和接触状态 |
| `joint_stiffness/damping/friction` | 随机化后的关节参数 |
| `body_mass/inertia/material/com` | 域随机化后的刚体参数 |

关键点：V14 的 actor policy 主要靠 `command + height_cmd + imu + joint state + last action`，不一定直接看实际 `base_lin_vel_b` 和 `obs_height`。这更接近 sim2real 常见设计：actor 不依赖难以稳定估计的真实线速度，高度也可能用额外通道/critic 处理。

### 当前 Wheel-Leg

当前 actor obs 布局在 `ObservationsCfg.PolicyCfg` 中写明：

```text
3 + 3 + 3 + 2 + 1 + 4 + 6 + 6 = 28 dims
history_length = 5
actor input = 140 dims
```

具体为：

| 项 | 维度 | 含义 |
|---|---:|---|
| `base_lin_vel` | 3 | base body-frame 线速度，scale `2.0` |
| `base_ang_vel` | 3 | base body-frame 角速度，scale `0.25` |
| `projected_gravity` | 3 | 重力方向投影，感知 roll/pitch |
| `velocity_commands` | 2 | `[vx_cmd, yaw_cmd]`，不放 `vy_cmd` |
| `height_command` | 1 | base height command，scale `5.0` |
| `joint_pos` | 4 | 四个腿部主动关节相对默认角 |
| `joint_vel` | 6 | 四腿关节 + 两轮关节速度，scale `0.05` |
| `last_action` | 6 | 上一帧动作 |

关键差别：

- 当前 actor 直接看到 `base_lin_vel`，对 vx 跟踪和原地抑制更直接。
- 当前 actor history 为 5 帧，输入实际是 140 维；V14 base policy dim 是 28，但也支持 frame stack。
- 当前项目没有 V14 那种复杂 privileged extra obs 和域随机化信息。

## Reward 对比

### 开源 V14 当前启用 reward

V14 在 `WheelbipeV14FlatEnvCfg` 中覆盖了 reward OrderedDict。下面是启用项和含义。

| reward | weight | 公式/含义 | 作用 |
|---|---:|---|---|
| `termination` | `-200` | episode 因失败终止时惩罚 | 强烈避免摔倒/非法终止 |
| `leg_joint_acc` | `-5e-7` | 腿部关节加速度平方和 | 抑制腿部高频加速度 |
| `leg_joint_vel` | `-5e-3` | 腿部关节速度平方和 | 抑制腿部快速摆动 |
| `leg_joint_pair_pos_diff` | `-0.0` | 左右腿配对关节位置差平方 | 当前关闭 |
| `joint_torque` | `-1e-4` | 受控关节力矩平方和 | 限制腿/轮输出力矩 |
| `wheel_acc` | `-1e-8` | 轮毂加速度平方和 | 弱抑制轮毂加速度 |
| `wheel_vel` | `-1e-5` | 轮毂速度平方和 | 弱抑制轮毂持续高速 |
| `wheel_power` | `-1e-4` | `max(tau * dq, 0)` 的轮毂耗功 | 限制轮毂正功率消耗 |
| `wheel_air_spin` | `0.0` | 轮子离地空转惩罚 | 当前关闭 |
| `lin_vel_z` | `-0.5` | `base_vz^2` | 抑制 base 上下跳动 |
| `ang_vel_xy` | `-0.05` | `wx^2 + wy^2` | 抑制 roll/pitch 角速度 |
| `action_smoothness_leg` | `-0.05` | 腿部 action 二阶差分平方 | 抑制腿 action 折返/抖动 |
| `action_rate` | `-0.01` | 全 action 一阶差分平方 | 抑制动作突变 |
| `action_smoothness_wheel` | `-0.01` | 轮毂 action 二阶差分平方 | 抑制轮毂高频反转 |
| `flat_orientation_y` | `-0.0` | `projected_gravity_b[x]` 平方 | 当前关闭 |
| `flat_orientation_y_v` | `-2.0` | 速度相关的 y 向姿态惩罚 | 随速度调节姿态约束 |
| `flat_orientation_y_exp` | `+1.0` | `exp(-pgb_x^2 / sigma)` | 奖励 roll/pitch 中一个方向接近水平 |
| `flat_orientation_x` | `-0.0` | `projected_gravity_b[y]` 平方 | 当前关闭 |
| `flat_orientation_x_v` | `-2.0` | 速度相关的 x 向姿态惩罚 | 随速度调节 pitch/roll 约束 |
| `flat_orientation_x_exp` | `+1.0` | `exp(-pgb_y^2 / sigma)` | 奖励另一个姿态方向接近水平 |
| `track_lin_vel_xy` | `+1.0` | `exp(-(vx_cmd - vx_horizontal)^2 / sigma)` | 主线速度跟踪 |
| `track_lin_vel_xy_tight` | `0.0` | 更窄 sigma 的速度跟踪 | 当前关闭 |
| `track_lin_vel_xy_square` | `-1.0` | `(vx_cmd - vx_horizontal)^2` | 直接惩罚速度误差 |
| `track_ang_vel_z` | `+1.0` | `exp(-(yaw_cmd - wz)^2 / sigma)` | yaw rate 跟踪 |
| `track_ang_vel_z_square` | `-1.0` | `(yaw_cmd - wz)^2` | 直接惩罚 yaw rate 误差 |
| `stand_still_lin_vel` | `-1.0` | stand deadzone 内惩罚 `abs(vxy)` | 站立命令下抑制平移 |
| `stand_still` | `-0.0` | stand deadzone 内惩罚 `vxy^2 + wz^2` | 当前关闭 |
| `track_height_exp` | `0.0` | `exp(-height_err^2 / sigma)` | 当前关闭 |
| `track_height_exp_soft` | `0.0` | 宽 sigma 高度跟踪 | 当前关闭 |
| `track_height_exp_tight` | `+1.0` | 窄 sigma 高度跟踪 | 主要高度精跟踪 |
| `track_height_square` | `-1.0` | `(height_err * height_square_sigma)^2` | 直接惩罚高度误差 |
| `track_height_exp_both_wheels_contact` | `0.0` | 双轮接触时高度奖励 | 当前关闭 |
| `no_fork` | `-1.0` | 左右轮 base x 距离超过阈值后惩罚 | 防止左右腿前后劈叉 |
| `no_fork_square` | `-1.0` | 左右轮 base x 差平方 | 连续惩罚劈叉程度 |
| `no_fork_exp` | `-0.0` | 超过阈值后的指数 no-fork 惩罚 | 当前关闭 |
| `no_fork_z_exp` | `-0.0` | 左右轮 z 差的指数惩罚 | 当前关闭 |
| `undesired_contact` | `-2.0` | 非期望 body 接触地面 | 防止机身/其他部件触地 |

V14 reward 的几个特点：

- 速度跟踪同时用了 `exp reward` 和 `square penalty`，这和我们后来给当前项目加 `lin_vel_tracking_error` 的方向一致。
- 高度跟踪也同时用了 `exp tight reward` 和 `square penalty`，不是只靠一个很窄的指数奖励。
- 对轮毂高频不是单纯惩罚 `wheel_vel`，还专门有 `action_smoothness_wheel`，这一点对抑制轮毂来回抖动很重要。
- 有 `no_fork/no_fork_square`，对应我们当前的 `wheel_x_alignment` 和 `wheel_x_separation termination`。
- V14 有 `stand_still_lin_vel`，但它依赖 stand deadzone，而不是全程把轮子锁死。

### 当前 Wheel-Leg 启用 reward

当前 `RewardsCfg` 的主要项如下。

| reward | weight | 含义 | 对应关系 |
|---|---:|---|---|
| `tracking_lin_vel` | `+3.0` | `exp(-(vx_cmd-vx)^2/sigma)` | 对应 V14 `track_lin_vel_xy` |
| `tracking_lin_vel_enhance` | `+1.0` | 宽松版 vx 跟踪，给大误差学习信号 | 类似 V14 soft tracking 思路 |
| `lin_vel_tracking_error` | `-4.0` | 直接惩罚 vx tracking error | 对应 V14 `track_lin_vel_xy_square` |
| `tracking_ang_vel` | `+2.0` | yaw rate 指令指数跟踪 | 对应 V14 `track_ang_vel_z` |
| `tracking_ang_vel_enhance` | `+1.0` | 宽松版 yaw tracking | V14 有 soft 版本但当前 V14 配置未启用 |
| `straight_yaw_rate` | `-5.0` | 无 yaw command 时惩罚实际 yaw rate | 当前项目新增，用来解决直线自转 |
| `straight_wheel_diff` | `-0.5` | 无 yaw command 时惩罚左右轮差速 action | 当前项目新增，用来辅助直线 |
| `base_height_coarse` | `+1.0` | 宽 sigma 高度跟踪 | V14 `track_height_exp_soft` 的思想 |
| `base_height` | `+2.0` | Fudan 风格窄 sigma 高度精跟踪 | 对应 V14 `track_height_exp_tight` |
| `stationary_vel` | `-1.0` | 无 vx command 时惩罚 base 水平速度 | 对应 V14 `stand_still_lin_vel` |
| `com_alignment` | `-0.5` | 无 vx command 时约束 COM x 接近左右轮中点 | 当前项目新增，针对高姿态后漂 |
| `root_position` | `-0.2` | 无 vx command 时限制长期离开 reset 原点 | 当前项目新增，防长期漂移 |
| `wheel_x_alignment` | `-0.2` | 左右轮 base x 方向不要错太多 | 对应 V14 `no_fork/no_fork_square` |
| `nominal_state` | `-5.0` | 左右虚拟腿摆角对称 | 当前项目腿型先验 |
| `mirrored_leg_pattern` | `0.0` | 关节镜像 `[+,+,-,-]` 先验 | 当前关闭 |
| `leg_vertical` | `-0.3` | 根据 height command 约束 hip->wheel 垂直支撑长度 | 当前项目新增，帮助站高 |
| `lin_vel_z` | `-0.1` | 惩罚 base z 速度 | 对应 V14 `lin_vel_z`，但当前更弱 |
| `ang_vel_xy` | `-0.2` | 惩罚 roll/pitch 角速度 | 对应 V14 `ang_vel_xy` |
| `orientation` | `-100.0` | 惩罚 roll/pitch 偏离竖直，不管 yaw | 对应 V14 flat orientation 系列 |
| `dof_vel` | `-5e-5` | 腿部关节速度平方 | 对应 V14 `leg_joint_vel`，但当前更弱 |
| `dof_acc` | `None` | 关节加速度惩罚关闭 | V14 启用 `leg_joint_acc` |
| `torques` | `-1e-4` | 受控关节力矩平方 | 对应 V14 `joint_torque` |
| `action_rate` | `-0.5` | 全 action 一阶差分 | 对应 V14 `action_rate`，当前强很多 |
| `action_smooth` | `-1.0` | 腿部 action 二阶差分 | 对应 V14 `action_smoothness_leg`，当前强很多 |
| `vx_wheel_antiphase` | `-0.05` | 有 vx command 且无 yaw command 时鼓励轮子反相 | 当前项目新增，辅助轮驱前后运动 |
| `collision` | `-1.0` | base link 接触惩罚 | 对应 V14 `undesired_contact` |
| `dof_pos_limits` | `-1.0` | 腿关节软限位惩罚 | V14 有 joint limit 备选项 |

当前项目和 V14 的主要 reward 差异：

| 类别 | V14 | 当前 Wheel-Leg |
|---|---|---|
| 速度跟踪 | exp + square penalty | exp + enhance + direct error penalty，权重更激进 |
| yaw 跟踪 | exp + square penalty | exp + enhance + 无 yaw 时额外压 yaw rate |
| 高度跟踪 | tight exp + square penalty | coarse exp + fine exp + leg_vertical |
| 原地站立 | `stand_still_lin_vel` | `stationary_vel + root_position + com_alignment` |
| 腿形/劈叉 | `no_fork/no_fork_square` | `wheel_x_alignment + termination + leg symmetry` |
| 动作平滑 | leg/wheel 分开二阶 smooth | 当前二阶 smooth 只压腿，wheel 主要通过 action_rate |
| reset 鲁棒性 | 随机腿长、腿角、轮角、关节速度、域随机化 | reset 更保守，关节回默认，root 小扰动 |

一个比较重要的点：V14 当前配置里 `action_smoothness_wheel=-0.01` 是 wheel-specific 二阶平滑；当前项目目前没有单独 wheel-specific 二阶 action smooth，只是 `action_rate` 对所有 action 生效。若后续轮毂高频反转严重，可以考虑参考 V14 增加很小的 wheel-specific 二阶平滑，而不是直接惩罚 `wheel_vel^2`。

## Reset 对比

### 开源 V14 / 25v3 reset

基础 reset 流程：

1. 清空 policy/critic history。
2. command generator reset。
3. 重新采样 height command。
4. root 和 joint 先写默认状态。
5. 调用 `_custom_reset_random(env_ids)` 做定制随机化。
6. 可选重采样 obs/action delay。

主要随机化：

| reset/randomization | 含义 |
|---|---|
| `use_leg_random_start=True` | reset 时随机腿长和腿角 |
| `leg_length_range=[0.2,0.32]` | 随机虚拟腿长 |
| `leg_angle_range=[-0.25*pi,0.75*pi]` | 随机虚拟腿摆角 |
| `wheel_angle_range=[-2*pi,2*pi]` | 轮子初始角随机 |
| `use_joint_vel_random_start=True` | reset 时给关节初速度 |
| `leg_joint_vel_range=[-pi/2,pi/2]` | 腿关节初速度随机 |
| `wheel_joint_vel_range=[-50,50]` | 轮子初速度随机 |
| `predefined_reset_ground` | 可选指定腿高/腿长/起始 root 高度/短时间 command override |
| `spring_settings` | 弹簧力和阻尼可随机 |
| startup 域随机化 | base/leg/wheel 质量、COM、摩擦、关节摩擦等 |

V14 的 `predefined_reset_ground` 中还有更接近课程学习的设定：

```text
positive mode: prob=0.3, sign=+1, leg_height=[-0.06,0.12], leg_length=[0.14,0.36]
negative mode: prob=0.2, sign=-1, leg_height=[-0.06,0.0], leg_length=[0.14,0.36]
start_reset_time=1.5
start_root_height=0.25
zero_torque_time_s=0.2
command_ranges: vx [-1,1], vy 0, yaw [-1,1]
```

这说明它不是简单“站好再开始”，而是通过很多随机初态让策略学会从不同腿型/轮速/姿态附近恢复。

### 当前 Wheel-Leg reset

当前 reset 更保守：

| 项 | 当前设置 | 含义 |
|---|---|---|
| root x/y | `[-0.05,0.05]` | reset 平面位置小扰动 |
| root z | `INITIAL_BASE_HEIGHT + [-0.02,0.02]` | base 初始高度小扰动 |
| root yaw | `[-0.1,0.1]` | 初始 yaw 小扰动 |
| root velocity | x/y `[-0.05,0.05]`, z `[-0.02,0.02]`, rpy rate `[-0.02,0.02]` | 初始速度很小 |
| joint pos | `(0.0,0.0)` offset | 关节回默认位 |
| joint vel | `(0.0,0.0)` | 关节速度清零 |

termination：

| termination | 含义 |
|---|---|
| `time_out` | 20s 正常结束 |
| `base_contact` | base link 碰地且过宽限步数后 reset |
| `low_base_height` | base 高度长期低于阈值后 reset |
| `leg_joint_deviation` | 腿关节偏离默认过大后 reset |
| `wheel_x_separation` | 左右轮 base x 劈叉过大后 reset |
| `bad_orientation` | roll/pitch 接近翻倒并持续后 reset |
| `root_state_non_finite` | 状态 NaN/Inf 后 reset |

当前 reset 的优点是稳定、易 debug、对 checkpoint 影响小；缺点是比 V14 的 reset 多样性弱，策略可能更依赖单一初始姿态。

## Command 对比

### 开源 V14

基础 25v3 里 velocity command：

```text
resampling_time_range = (1.0, 8.0)
rel_standing_envs = 0.1
rel_heading_envs = 0.5
heading_command = True
lin_vel_x = [-3.0, 3.0]
lin_vel_y = [0.0, 0.0]
ang_vel_z = [0.0, 0.0]
heading = [0.0, 0.0]
```

height command：

```text
default_height_cmd = 0.25
height_range = [0.22, 0.35]
```

V14/cfg_utils 的 predefined reset ground 还会在 reset 初期临时覆盖 command range 到：

```text
vx [-1, 1], vy 0, yaw [-1, 1]
```

### 当前 Wheel-Leg

当前 command：

```text
BASE_HEIGHT_COMMAND_RANGE = (0.3, 0.42)
COMMAND_RESAMPLING_TIME = 5.0
BASE_HEIGHT_RESAMPLING_TIME = 100.0
COMMAND_LIN_VEL_X_RANGE = (-1.0, 1.0)
COMMAND_LIN_VEL_Y_RANGE = (0.0, 0.0)
COMMAND_ANG_VEL_Z_RANGE = (-8.0, 8.0)
COMMAND_STANDING_ENV_RATIO = 0.15
COMMAND_YAW_ENV_RATIO = 0.0
COMMAND_FORWARD_VX_ENV_RATIO = 0.5
```

注意：当前 `base_height` command 使用独立 `BASE_HEIGHT_RESAMPLING_TIME=100.0`，因此一个 20s episode 内通常不切换高度；velocity command 仍按 5s 重采样。

## 对当前训练的启发

1. 速度跟踪方面，V14 的思路是 `exp tracking + square error penalty`。当前项目已经采用类似结构，而且权重更强，所以如果仍然学不好 vx/yaw，问题可能不只是 reward 形式，也可能是 action 映射、命令范围、轮毂动力学、姿态/高度 reward 冲突。

2. 高度方面，V14 同时用了 `track_height_exp_tight` 和 `track_height_square`。当前项目用 `coarse + fine + leg_vertical`，对高姿态已经更定制。若高姿态仍卡住，可以检查 `leg_vertical`、`orientation`、`COM alignment` 是否互相拉扯。

3. 原地平衡方面，V14 更简单，用 `stand_still_lin_vel`；当前项目更复杂，用 `stationary_vel + root_position + com_alignment`。这更适合诊断“后漂”和“COM 后偏”，但也更容易和速度跟踪发生冲突，所以 conditional gate 很关键。

4. 轮毂抖动方面，V14 有 `action_smoothness_wheel`。当前如果后面继续出现轮毂高频反转，更推荐加一个小权重 wheel-specific 二阶 action smooth，而不是加大 `wheel_vel` 或 `stationary_vel`。

5. reset 方面，V14 的 reset 多样性明显更强。当前先用保守 reset 练出基本能力是合理的；等稳定后，可以逐步加入小幅 leg pose / wheel vel / push / 摩擦随机化，而不是一开始全部打开。

## 可迁移优先级建议

当前不建议“完全照搬 V14”。更建议按优先级迁移：

1. 保留当前 actor obs，因为已经加入 `base_lin_vel` 且更利于速度跟踪。
2. 保留当前高度 reward 结构，不要退回只用 V14 的 narrow height reward。
3. 可参考 V14 增加 `wheel_action_smoothness`，只作用在 wheel action 二阶差分，权重从 `-0.005` 到 `-0.01` 试。
4. reset 后期可以参考 V14，逐步加入轻量随机初态：轮子初速度、腿部小角度扰动、push、摩擦随机化。
5. velocity/yaw command range 需要课程化。当前 yaw `[-8,8]` 比 V14 大很多，若 yaw 学不好，先不要只加 reward，应该同步检查可达 yaw rate 和 command curriculum。

