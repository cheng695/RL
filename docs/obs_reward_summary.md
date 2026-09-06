# Wheel-Leg 当前 Observation / Reward 总结

来源文件：

- `Wheel-Leg/envs/wheel_leg/wheel_leg_env_cfg.py`
- `Wheel-Leg/envs/wheel_leg/mdp/rewards.py`
- `Wheel-Leg/envs/wheel_leg/mdp/commands.py`

## Observation

当前策略 observation group 为 `policy`，`concatenate_terms=True`，所以所有 term 会按顺序拼成一个向量。`enable_corruption=False`，当前没有 observation noise/corruption。

| 顺序 | 名称 | 函数 | 维度 | 内容 |
|---:|---|---|---:|---|
| 1 | `base_ang_vel` | `mdp.base_ang_vel` | 3 | base 在机体系下的角速度，通常为 roll/pitch/yaw rate |
| 2 | `base_pos_z` | `mdp.base_pos_z` | 1 | base/root 的世界系 z 高度 |
| 3 | `projected_gravity` | `mdp.projected_gravity` | 3 | 重力方向投影到机体系后的向量 |
| 4 | `velocity_commands` | `mdp.generated_commands("base_velocity")` | 3 | 速度命令 `[vx, vy, yaw_rate]` |
| 5 | `height_command` | `mdp.generated_commands("base_height")` | 2 | 高度命令 `[target_height, target_height_rate]` |
| 6 | `joint_pos` | `mdp.joint_pos_rel` | 6 | 6 个受控关节相对 default 的位置 |
| 7 | `joint_vel` | `mdp.joint_vel_rel` | 6 | 6 个受控关节相对 default 的速度 |
| 8 | `last_action` | `mdp.last_action` | 6 | 上一步 policy 输出的 6 维 action |

总维度：`3 + 1 + 3 + 3 + 2 + 6 + 6 + 6 = 30`

这里的 `joint_pos` / `joint_vel` 是 observation 里读到的关节状态，不是 policy 直接输出。policy 直接输出的是 6 维 action，顺序与受控关节一致：

1. `Left_front_joint`
2. `Left_rear_joint`
3. `Right_front_joint`
4. `Right_rear_joint`
5. `Left_Wheel_joint`
6. `Right_Wheel_joint`

## Policy 输出 Action

当前 policy 输出 6 维 action：

| action 维度 | 对应关节 | 控制类型 | 尺度 | 实际含义 |
|---:|---|---|---:|---|
| 0 | `Left_front_joint` | 位置控制 | `0.8 rad` | 目标角 = USD/default 角 + action[0] * 0.8 |
| 1 | `Left_rear_joint` | 位置控制 | `0.8 rad` | 目标角 = USD/default 角 + action[1] * 0.8 |
| 2 | `Right_front_joint` | 位置控制 | `0.8 rad` | 目标角 = USD/default 角 + action[2] * 0.8 |
| 3 | `Right_rear_joint` | 位置控制 | `0.8 rad` | 目标角 = USD/default 角 + action[3] * 0.8 |
| 4 | `Left_Wheel_joint` | 速度控制 | `30.0 rad/s` | 目标轮速 = action[4] * 30.0 |
| 5 | `Right_Wheel_joint` | 速度控制 | `30.0 rad/s` | 目标轮速 = action[5] * 30.0 |

也就是说：policy 不是直接输出 obs，而是输出「四个腿关节位置目标 + 两个轮子速度目标」。`clip_actions=1.0` 时，action 会被裁剪到 `[-1, 1]`。腿关节目标最多偏离 USD/default 约 `±0.8 rad`，action term 内部会对最终目标角再做 `default ±0.8 rad` 硬限幅，并限制每个 policy step 目标角最多变化 `0.08 rad`；轮子目标速度最多约 `±30 rad/s`。

## Command 和 Action 尺度

| 项 | 当前值 | 说明 |
|---|---:|---|
| `COMMAND_LIN_VEL_X_RANGE` | `(0.0, 0.0)` | vx 命令固定为 0 |
| `COMMAND_LIN_VEL_Y_RANGE` | `(0.0, 0.0)` | vy 命令固定为 0 |
| `COMMAND_ANG_VEL_Z_RANGE` | `(0.0, 0.0)` | yaw rate 命令固定为 0 |
| `COMMAND_STANDING_ENV_RATIO` | `1.0` | 全部 env 使用站立速度命令 |
| `BASE_HEIGHT_COMMAND_RANGE` | `(0.35, 0.45)` | 目标高度范围为 0.35~0.45 m |
| `BASE_HEIGHT_COMMAND_RATE_LIMIT` | `0.0` | 高度命令变化速度为 0 |
| `BASE_HEIGHT_COMMAND_DELTA_RANGE` | `(-0.01, 0.01)` | 旧随机游走参数，当前三角波主要使用 rate limit |
| `LEG_POSITION_ACTION_SCALE` | `0.8` | 腿关节目标角 = default 角 + action * 0.8 rad，并硬限幅到 default ±0.8 rad |
| `LEG_POSITION_MAX_TARGET_DELTA` | `0.08` | 腿部真实关节目标每个 policy step 最大变化量，单位 rad |
| `WHEEL_VELOCITY_ACTION_SCALE` | `30.0` | 轮子目标速度 = action * 30.0 rad/s |
| `clip_actions` | `1.0` | RSL-RL runner 中 action 裁剪到 `[-1, 1]` |

## Reward 总览

总 reward 由 Isaac Lab reward manager 按 `weight * term_value` 求和。正权重为奖励，负权重为惩罚。

| 名称 | 权重 | 参数 | 含义 |
|---|---:|---|---|
| `tracking_lin_vel` | `0.2` | `tracking_sigma=0.25` | 奖励 base x 方向速度跟踪 `vx_cmd`，形式为 `exp(-(vx_cmd - vx)^2 / tracking_sigma)` |
| `tracking_ang_vel` | `0.5` | `tracking_sigma=0.25` | 奖励 yaw rate 跟踪，形式为 `exp(-(yaw_cmd - yaw_rate)^2 / tracking_sigma)` |
| `base_height` | `1.0` | `tracking_sigma=0.001` | 奖励 base 高度接近目标高度：`exp(-(z - height_cmd)^2 / 0.001)` |
| `base_height_l2` | `-5.0` | `command_name=base_height` | 温和惩罚 base 高度偏离目标高度：`(z - height_cmd)^2` |
| `lin_vel_z` | `-2.0` | 无 | 惩罚 base z 方向线速度平方：`base_lin_vel_z^2` |
| `ang_vel_xy` | `-0.05` | 无 | 惩罚 base roll/pitch 角速度平方和：`base_ang_vel_x^2 + base_ang_vel_y^2` |
| `orientation` | `-10.0` | 无 | 惩罚机身倾斜：`projected_gravity_x^2 + projected_gravity_y^2` |
| `yaw` | `-0.5` | 无 | 临时惩罚 base yaw 角平方，抑制固定命令下原地转圈 |
| `root_xy_position` | `-5.0` | `deadband=0.05` | 惩罚 base 相对 env 原点的水平漂移，超过 0.05 m 后按平方惩罚 |
| `base_contact` | `-20.0` | `threshold=0.1 N` | base_link 接触力超过阈值时返回 1 并惩罚 |
| `wheel_air` | `-8.0` | `threshold=1.0 N` | 任意轮子接触力低于阈值，即轮子离地时返回 1 并惩罚 |
| `wheel_x_separation` | `-20.0` | `deadband=0.02 m` | 惩罚左右轮在 base x 方向错开，超过 0.02 m 后按平方惩罚 |
| `leg_joint_vel` | `-0.035` | joints=4 个腿关节 | 惩罚腿关节速度平方和 |
| `leg_joint_acc` | `-3.0e-5` | joints=4 个腿关节 | 惩罚腿关节加速度平方和 |
| `action_rate` | `-0.15` | all actions | 惩罚 action 相邻两步变化量平方和 |

## 当前奖励/惩罚系数

主要奖励：

| 系数 | 值 |
|---|---:|
| 速度跟踪 `tracking_lin_vel` | `0.2` |
| yaw rate 跟踪 `tracking_ang_vel` | `0.5` |
| 高度跟踪 `base_height` | `1.0` |

主要惩罚：

| 系数 | 值 |
|---|---:|
| 高度误差 `base_height_l2` | `-5.0` |
| z 方向速度 `lin_vel_z` | `-2.0` |
| roll/pitch 角速度 `ang_vel_xy` | `-0.05` |
| 机身倾斜 `orientation` | `-10.0` |
| yaw 角 `yaw` | `-0.5` |
| 水平漂移 `root_xy_position` | `-5.0` |
| base 接触地面 `base_contact` | `-20.0` |
| 轮子离地 `wheel_air` | `-8.0` |
| 左右轮 x 错开 `wheel_x_separation` | `-20.0` |
| 腿关节速度 `leg_joint_vel` | `-0.035` |
| 腿关节加速度 `leg_joint_acc` | `-3.0e-5` |
| action rate `action_rate` | `-0.15` |

## 相关阈值和限制

| 名称 | 当前值 | 用途 |
|---|---:|---|
| `TARGET_BASE_HEIGHT` | `0.35 m` | 初始/最低目标高度 |
| `TRACKING_SIGMA` | `0.25` | 速度/yaw rate 跟踪 reward 的指数衰减系数 |
| `BASE_HEIGHT_TRACKING_SIGMA` | `0.001` | base 高度跟踪 reward 的指数衰减系数 |
| `BASE_CONTACT_REWARD_FORCE_THRESHOLD` | `0.1 N` | base_link 接触 reward 判定阈值 |
| `BASE_CONTACT_TERMINATION_FORCE_THRESHOLD` | `10.0 N` | base_link 接触 termination 判定阈值 |
| `WHEEL_CONTACT_FORCE_THRESHOLD` | `1.0 N` | 轮子接触/离地判定阈值 |
| `WHEEL_X_SEPARATION_REWARD_DEADBAND` | `0.02 m` | 左右轮 x 错开惩罚 deadband |
| `ROOT_XY_POSITION_REWARD_DEADBAND` | `0.05 m` | 水平漂移惩罚 deadband |
| `LIN_VEL_Z_REWARD_CLIP` | `25.0` | 传入 `lin_vel_z_l2`，但当前函数未实际 clamp |
| `ANG_VEL_XY_REWARD_CLIP` | `100.0` | 传入 `ang_vel_xy_l2`，但当前函数未实际 clamp |
| `JOINT_VEL_REWARD_CLIP` | `1.0e4` | 传入 `joint_vel_l2`，但当前函数未实际 clamp |
| `JOINT_ACC_REWARD_CLIP` | `1.0e8` | 传入 `joint_acc_l2`，但当前函数未实际 clamp |
| `ACTION_RATE_REWARD_CLIP` | `1.0e3` | 传入 `action_rate_l2`，但当前函数未实际 clamp |

## 小提醒

`lin_vel_z_l2`、`ang_vel_xy_l2`、`joint_vel_l2`、`joint_acc_l2`、`action_rate_l2` 虽然接收了 `max_value` 参数，但函数实现里目前只是直接返回平方和，没有做裁剪。
