# Wheel-Legged-Gym Reward Summary

路径：`/home/whc/桌面/RL/Wheel-Legged-Gym-master`

`wheel_legged_vmc_flat` 继承关系：

```text
WheelLeggedVMCFlatCfg
  -> WheelLeggedVMCCfg
  -> WheelLeggedCfg
  -> LeggedRobotCfg
```

当前任务没有在 `wheel_legged_vmc_flat_config.py`、`wheel_legged_vmc_config.py` 或 `wheel_legged_config.py` 里覆盖 reward scales，所以实际使用的是 base 配置：

```text
Wheel-Legged-Gym-master/wheel_legged_gym/envs/base/legged_robot_config.py
```

reward 实现函数在：

```text
Wheel-Legged-Gym-master/wheel_legged_gym/envs/base/legged_robot.py
```

## Reward 计算方式

环境会读取 `cfg.rewards.scales` 中所有非零项，并查找对应函数：

```text
reward name -> self._reward_<reward name>()
```

例如：

```text
tracking_lin_vel -> self._reward_tracking_lin_vel()
```

每个非零权重会先乘以 `dt`：

```text
effective_scale = scale * dt
```

每个单项 reward 会被裁剪到：

```text
[-clip_single_reward * dt, clip_single_reward * dt]
```

当前配置：

```text
only_positive_rewards = False
clip_single_reward = 1
tracking_sigma = 0.25
soft_dof_pos_limit = 0.97
soft_dof_vel_limit = 1.0
soft_torque_limit = 1.0
base_height_target = 0.18
max_contact_force = 100.0
```

## 当前启用 Reward

| reward | 权重 | 作用 | 实现/形式 |
|---|---:|---|---|
| `tracking_lin_vel` | `1.0` | 奖励 base x 方向速度跟踪命令 | `exp(-(vx_cmd - vx)^2 / tracking_sigma)` |
| `tracking_lin_vel_enhance` | `1.0` | 额外速度跟踪 shaping | `exp(-(vx_cmd - vx)^2 / tracking_sigma / 10) - 1` |
| `tracking_ang_vel` | `1.0` | 奖励 yaw 角速度跟踪命令 | `exp(-(yaw_cmd - yaw_rate)^2 / tracking_sigma)` |
| `base_height` | `1.0` | 奖励 base 高度接近命令高度 | 正权重时 `exp(-(base_height - height_cmd)^2 / 0.001)` |
| `nominal_state` | `-0.1` | 惩罚左右腿虚拟角不一致 | `(theta0_left - theta0_right)^2` |
| `lin_vel_z` | `-2.0` | 惩罚 base z 方向线速度 | `base_lin_vel_z^2` |
| `ang_vel_xy` | `-0.05` | 惩罚 base roll/pitch 角速度 | `base_ang_vel_x^2 + base_ang_vel_y^2` |
| `orientation` | `-10.0` | 惩罚机身倾斜 | `projected_gravity_x^2 + projected_gravity_y^2` |
| `dof_vel` | `-5e-5` | 惩罚腿关节速度 | 左右腿关节速度平方和，不含轮子 |
| `dof_acc` | `-2.5e-7` | 惩罚关节加速度 | 全部 dof 加速度平方和 |
| `torques` | `-0.0001` | 惩罚电机力矩 | 全部 torque 平方和 |
| `action_rate` | `-0.01` | 惩罚 action 一阶变化 | `(last_action - action)^2` |
| `action_smooth` | `-0.01` | 惩罚 action 二阶变化 | `action - 2 * last_action + last_last_action` 的平方 |
| `collision` | `-1.0` | 惩罚指定 body 接触 | penalized contact body 接触力超过 `0.1` |
| `dof_pos_limits` | `-1.0` | 惩罚腿关节超过位置限制 | 只检查腿关节，不含轮子 |

## 默认未启用 Reward

这些 reward 函数在 `legged_robot.py` 中存在，但默认 `cfg.rewards.scales` 没有给非零权重，所以不会被 reward manager 调用。

| reward | 作用 |
|---|---|
| `base_height_enhance` | 高度跟踪 shaping |
| `power` | 惩罚 `abs(torque * dof_vel)` |
| `dof_vel_limits` | 惩罚接近/超过关节速度软限制 |
| `torque_limits` | 惩罚接近/超过 torque 软限制 |
| `tracking_ang_vel_enhance` | yaw 角速度跟踪 shaping |
| `tracking_lin_vel_pbrs` | x 速度跟踪 potential-based reward shaping |
| `tracking_ang_vel_pbrs` | yaw 角速度跟踪 potential-based reward shaping |
| `stumble` | 惩罚脚/轮撞到竖直面 |
| `stand_still` | 低速命令时惩罚偏离默认关节姿态 |
| `feet_contact_forces` | 惩罚脚/轮接触力超过 `max_contact_force` |
| `termination` | reset 时的终止奖励/惩罚 |

## 备注

`base_height` 的实现会根据权重正负改变形式：

```text
如果 base_height 权重 < 0:
    abs(base_height - height_cmd)
否则:
    exp(-(base_height - height_cmd)^2 / 0.001)
```

当前权重是 `1.0`，所以它是正奖励形式。

`dof_vel` 和 `dof_pos_limits` 主要针对腿关节：

```text
dof_vel: dof[:, :2] + dof[:, 3:5]
dof_pos_limits: dof[:, :2] + dof[:, 3:5]
```

也就是左右腿关节，不包含两个轮子关节。
