# UZ05 左右腿质量对称实验

## 实验目的与默认选择

检查左右腿质量差异是否影响负方向转向、pitch/roll 和轮速振荡。
这些现象可能同时受控制、接触和随机质心偏移影响；尚未通过本实验确认因果关系。

当前默认选择 **equalized** 版本。设置 `UZ05_MASS_VARIANT=original` 可恢复原始模型，
设置 `UZ05_MASS_VARIANT=equalized` 可显式选择实验模型；启动日志会打印选中的源文件。
每次更改选择后重新启动进程。

PLAY 现在会读取 checkpoint 旁的 `params/env.yaml`，默认匹配其质量版本与机身质心随机化设置；
显式设置 `UZ05_MASS_VARIANT` 或 `UZ05_DISABLE_BASE_COM` 时，以对应环境变量为准。
这可避免训练时关闭质心随机化、播放时却重新启用的问题，也避免旧策略意外加载实验质量。
没有配置快照的 checkpoint 使用当前配置并打印提示。最终生效设置以 PLAY 恢复日志为准，
新 CSV 的同名 JSON 也记录实际资产路径及质心随机化是否启用。

- 原始模型：`uz05_isaac.xml`，保留原文件。
- 实验模型：同目录下的 `uz05_isaac_equal_leg_masses.xml`。
- USD 缓存：分别为 `Wheel-Leg/assets/uz05`、`Wheel-Leg/assets/uz05_equal_leg_masses`。
- 此实验参数用于仿真对照；实际机器人质量有差异时，应保留真实参数作为最终验证模型。

## 修改方法

逐对取左右对应 link 的平均质量，各自惯量张量乘以 `新质量 / 原质量`。
质心局部位置、关节、几何和控制参数保留。这样保持每个 link 的单位质量惯量特性，
不会因为仅改质量而保留不匹配的原惯量。对称的是成对质量，不是将所有 link 设成相同质量。

| 对应部件 | 原左侧 kg | 原右侧 kg | 修改后每侧 kg |
|---|---:|---:|---:|
| link2 | 0.337265152 | 0.386232978 | 0.361749065 |
| link4 | 0.271215118 | 0.109162612 | 0.190188865 |
| link5 | 0.107885473 | 0.107885473 | 0.107885473 |
| link3 | 0.055275907 | 0.055275907 | 0.055275907 |
| link1 | 0.057077033 | 0.057077033 | 0.057077033 |
| link6 | 0.356789896 | 0.356789896 | 0.356789896 |
| wheel | 0.245116509 | 0.245116509 | 0.245116509 |
| **腿总成（含轮子）** | **1.430625088** | **1.317540408** | **1.374082748** |

整机名义总质量仍为 **15.8 kg**。link3/link6 等原有的极小浮点差异也按平均值统一。

MuJoCo 编译并在零位姿执行前向计算后，整机横向质心坐标从约 +1.444 mm 变为 0；
前后及竖直质心坐标保持一致。此数值说明名义静态横向偏移较小，
不能仅凭质量不对称就解释全部转向问题，仍需对照。

重新生成实验模型（无需启动 Isaac Sim）：

```bash
python Wheel-Leg/scripts/equalize_uz05_leg_masses.py
```

生成脚本读取原始文件，写入独立实验文件。原始质量修改后，需重新执行该脚本。

## 导入器修正

之前 `ClosedLoopMjcfConverter` 从固定的 `uz05.xml` 编译并回填 USD 惯性参数，
因此只修改 `uz05_isaac.xml` 的质量可能在导入后被覆盖。

现在仍使用 MuJoCo 原生参考模型的 frame/site 闭链拓扑，但先将所选导入 MJCF 的显式
inertial 定义按 body 名称复制到参考模型，再编译并回填质量、质心及主惯量。
这样实验模型的修改可以进入 USD，而原始版本的编译惯性结果保持一致。

已检查原始版回填前的编译质量、质心、主惯量和惯性轴与旧参考模型一致；
实验版左右对应质量相等、总质量保持 15.8 kg，有质量刚体的主惯量为正并满足惯量三角不等式。
这些为 CPU/MuJoCo 检查，尚未启动 Isaac Sim 验证 USD 导入及训练效果。

## 控制机身质心随机化

现有训练和 PLAY 配置会在启动时随机偏移机身质心：x/y ±0.05 m、z ±0.01 m。
这与左右腿名义质量差异是不同因素。对照时设置 **`UZ05_DISABLE_BASE_COM=1`**，
在两种模型上都关闭该项；默认不设置时仍按当前任务配置随机化。
该开关同时作用于训练和 PLAY。相同种子能减少随机差异，但不能替代记录实验配置。

## 同 checkpoint 快速对照

先对原始模型记录一次，再将环境变量和输出文件名替换为实验版。
比较 ±1.5 rad/s、同高度及种子的两方向结果；下例为负方向。

```bash
UZ05_MASS_VARIANT=original UZ05_DISABLE_BASE_COM=1 \
python Wheel-Leg/scripts/rsl_rl/play.py \
  --task MyRobot-Velocity-Flat-Play-v0 --num_envs 1 \
  --checkpoint "/home/whc/桌面/RL/logs/rsl_rl/uz05_flat/2026-10-20_20-36-09_yaw_tracking_continue/model_4900.pt" \
  --seed 42 --command_vx 0 --command_yaw -1.5 --command_height 0.30 \
  --no_keyboard --no_hud --headless \
  --motion_trace_csv logs/motion_traces/mass_original_yaw_neg15.csv \
  --motion_trace_steps 1000
```

实验版改为 `UZ05_MASS_VARIANT=equalized`，输出名例如
`logs/motion_traces/mass_equalized_yaw_neg15.csv`。正方向改 `--command_yaw 1.5` 并使用新的输出名。

旧策略在新动力学上的表现只能用于快速诊断。没有立即改善也不能排除质量因素，因为旧策略可能已经学会部分不对称补偿。

## 从头训练实验模型

当前从头训练使用低速优先分布：vx 三档 60/25/15，yaw 四档 60/25/10/5，
各档边界及总体模式比例见 [motion_trace.md](motion_trace.md)。
原始质量与对称质量两组应采用相同命令分布；不要直接将本轮与旧的中高速采样 run 比较并归因于质量。

```bash
UZ05_MASS_VARIANT=equalized UZ05_DISABLE_BASE_COM=1 \
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Flat-v0 --num_envs 1024 --seed 42 \
  --max_iterations 3000 --run_name mass_equalized --headless
```

严格训练对照使用相同设置，并仅将 variant 改为 `original`、run_name 改为 `mass_original`。
现有成功 run 若启用了质心随机化，则不构成此实验的唯一变量对照。
保存各 run 的 `params/env.yaml` 和代码版本；资产路径会标明实验文件，关闭质心随机化时 `events.base_com` 为 null。

评估 yaw MAE/偏差、roll/pitch、高度误差、最终目标变化幅度、估计力矩及失败次数，
同时复测 vx 正反方向，避免转向改善而直行退化。
