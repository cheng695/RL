## 当前任务：扩大速度与混合转向，提高地形难度 v3

训练指令范围：**vx -1～1 m/s，yaw -3～3 rad/s**。保留 20% 静止；其余指令
直行/原地转向/平移加旋转 = 20%/20%/60%，对应总采样比例约 16%/16%/48%。
保留轮速工作空间投影，较大的 vx/yaw 组合会同比缩小，不能保证同时达到两个范围上限。
例如请求 vx=1、yaw=3，按当前 20 rad/s 轮速工作空间，有效指令约为 vx=0.675、yaw=2.024。

地图仍为 30% 平地、35% 粗糙、35% 波浪，10 行 × 20 列。粗糙采样幅度从
±3 mm 随难度增加至 **±25 mm**（原上限 ±20 mm），采样间距仍为 0.25 m；
波浪幅度参数从 5 mm 增加至 **50 mm**（原上限 40 mm），每块仍为 2 个周期，
二维地面的峰谷差约 10～100 mm。插值和离散化会使实际值有所差异。
所有环境从第 0 级开始；Play 同步使用新地图，原 checkpoint 也会面对新地形。

课程只对直行移动回合执行原距离升降级判据，静止、原地转向和混合转向回合保持等级，
避免绕圈后净位移小被误判为失败。保留 v2 防劈叉、稳定性和失败奖励，以及 94 维观测、6 维动作。
Play 键盘范围仍为 vx ±1、yaw ±5，yaw 大于 3 仍超出本版训练范围。

从已有 v2 checkpoint 续训（可替换为你确认表现更好的 v2 模型）：

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Rough-Scan-v0 \
  --num_envs 1024 \
  --resume \
  --checkpoint "/home/whc/桌面/RL/logs/rsl_rl/uz05_rough_scan/2026-09-23_12-04-14_gentle_rough_balance_v2/model_5000.pt" \
  --reset_noise_std 0.08 \
  --max_iterations 3000 \
  --run_name rough_twist_terrain_v3 \
  --headless
```

下面保留历史参数；运行旧命令也会加载当前代码配置。

## 历史配置：轻度粗糙与起伏，稳定性奖励 v2

`MyRobot-Velocity-Rough-Scan-v0` 已改为 30% 平地、35% 随机粗糙、35% 平缓波浪，
不再使用台阶、方块、陡坡和爬阶状态机。10 行 × 20 列，每块 8×8 m；所有环境从第 0 级开始。
粗糙采样幅度随难度从 ±3 mm 增至 ±20 mm，采样间距 0.25 m，经插值生成地面
（插值可能轻微超出采样高度范围）。波浪每块 2 个周期，参数幅度 5～40 mm，
按本机生成器实现，整个二维地面的峰谷差约 10～80 mm。各行在对应难度区间内采样，
不是第 0 行严格等于幅度下限。平地列用于保留基础运动能力。

采用官方距离课程：离出生点超过 4 m 升级；未达到指令预期距离的一半则降级。
速度指令每 30 秒重采样，通常一个回合保持方向，避免中途往返影响净位移判据。
vx ±0.5 m/s、yaw ±1 rad/s；20% 静止，其余直行/转向/混合为 70%/10%/20%。
目标高度 0.30～0.36 m，继续使用局部地面参考。

奖励沿用平地 vx/yaw 跟踪和静止防漂；高度指数核系数 1000→300、高度误差权重 -1→-0.5，
竖直速度惩罚 -2→-0.5、腿部速度惩罚 -0.01→-0.005。姿态奖励权重 3，
roll 保持约束，pitch 按扫描起伏允许约 3～8 度；取消平地直行 pitch 辅助惩罚。
姿态正奖励权重已从 v1 的 2 提高到 3，并增加权重 -0.5 的超出俯仰容差惩罚。
轮子前后错位容差从 8 cm 收紧到 4 cm，尺度从 15 cm 调整为 10 cm，权重 -0.25→-1；
采用二次到线性的连续惩罚，去掉严重劈叉时的封顶。滚转/俯仰角速度惩罚 -0.05→-0.15，
腿部动作二阶平滑惩罚 -0.03→-0.04。每次物理失败额外扣 5 分（抵消 dt 缩放），
单纯超时不扣分。保留常规底盘接触、倾倒和腿长终止条件；
没有爬阶成功、抬轮或支撑奖励，也没有爬阶接触豁免。

本次调整针对 Play 中的劈叉和失稳反馈。v1 最后 100 轮日志：正常超时比例约 99.59%，
姿态奖励约 1.981/2，劈叉奖励约 -0.0178，平均地形等级约 5.65。
这些是训练聚合指标，不代表交互 Play 或每种地形都稳定；还需固定速度、转向和等级比较。
奖励数值因定义改变，不应直接比较新旧总回报。

已确认用户复现条件：`model_2999.pt`、波浪列 13、等级 0，vx=0.7 m/s、yaw=1～2 rad/s。
速度超出本任务训练范围（vx ±0.5、yaw ±1）。Play 键盘档位现按用户要求允许 vx ±1 m/s、yaw ±5 rad/s；
显式 `--command_vx/--command_yaw` 仍允许越界测试，但会输出警告。
先在训练范围内复测；若目标是 0.7 m/s、2 rad/s，需要另行扩展训练速度分布。
本版只修改奖励与 Play 档位约束，训练地形、速度范围及观测/动作维度保持不变。

v2 验证：39 项 CPU 测试通过；20 环境、80 步 GPU 零动作检查通过，
未启动完整续训，尚不能据此判断新策略稳定性。

从本次已完成的 94 维模型继续训练 v2：

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Rough-Scan-v0 \
  --num_envs 1024 \
  --resume \
  --checkpoint "/home/whc/桌面/RL/logs/rsl_rl/uz05_rough_scan/2026-09-23_10-26-41_gentle_rough_waves_v1/model_2999.pt" \
  --reset_noise_std 0.08 \
  --max_iterations 3000 \
  --run_name gentle_rough_balance_v2 \
  --headless
```

修改 reward 不会直接改变旧模型 Play 的动作，需续训后加载新 checkpoint。
以下保留从原始平地模型重新迁移的方式：

观测仍为 49 维原输入 + 45 维扫描，动作不变。首次从指定平地模型使用
`--transfer_from`，自动复制旧输入权重、新增扫描权重置零，并重建优化器。
这是网络输入扩展的等价初始化；地形、高度参考和指令范围变化仍需要重新训练验证。

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Rough-Scan-v0 \
  --num_envs 1024 \
  --transfer_from "/home/whc/桌面/RL/logs/rsl_rl/uz05_flat/2026-09-20_12-35-39_twist_workspace_v8/model_20294.pt" \
  --reset_noise_std 0.10 \
  --max_iterations 3000 \
  --run_name gentle_rough_balance_v2_from_flat \
  --headless
```

训练后的 94 维模型用以下命令检查；列 0～5 为平地、6～12 为粗糙、13～19 为波浪，
等级 0～9。Play 默认最低级并冻结课程，与训练使用同一张地图。

```bash
python Wheel-Leg/scripts/rsl_rl/play.py \
  --task MyRobot-Velocity-Rough-Scan-Play-v0 \
  --num_envs 1 \
  --terrain_column 13 --terrain_level 0 \
  --checkpoint "/替换为本次训练目录/model_XXX.pt" \
  --real_time
```

继续训练本版模型才使用 `--resume --checkpoint`。观察 vx/yaw 跟踪误差、height_mae_cm、
存活时间和 terrain_levels，并逐级 Play 验证；爬阶通过率指标不再适用于本版任务。
下面是历史配置记录，其中旧 Rough-Scan 命令如今也会加载新地图，不能复现旧台阶任务。

v1 验证记录：36 项 CPU 回归测试通过；20 个环境、50 步 GPU 零动作冒烟检查通过，
观测 `(20, 94)`、动作 `(20, 6)`，奖励与观测均为有限值。未启动完整训练，
这不代表已学会粗糙地形。仿真仍报告 MJCF spatial tendon attachment 拓扑警告，
涉及 wheel/hip axis limit site；本次未修改机器人模型。

## 历史配置：官方 Rough 框架 + 两级台阶

上下行楼梯替换为仅两级的方环台阶：中心平台3m，中间踏面0.8m，第二级后为平台，
不再在同一地形块内连续爬楼梯。其他地形比例不变，训练和Play一致。
第一级固定抬升150mm，第二级再抬升200mm，顶部累计350mm；所有地形等级均保持这两个高度。
下行区域从中心向外依次下降150mm、200mm。每级稳定通过奖励8分，两级全部完成额外8分；
同一回合退回后重复爬同一高度不再给分。
训练完成两级后按成功截断reset，避免随后驶出地形块的摔倒污染通过率；Play完成后可继续观察，
仍保留摔倒/超时自动reset。旧checkpoint可加载，但会面对新地图，需续训验证。

爬阶 reward 修正：成功要求两轮越过首次台阶顶面采样位置，轮底高度容差为台阶高度的25%
（限5～25mm），避免下层或腾空误判成功。爬阶期间关闭持续的高度正奖励，高度误差惩罚乘0.1；
最远前进增量奖励4分/m，按落后轮的位置计分，单纯前倾不能增加此项。
靠近阶沿后，两轮中较低轮的抬升新进展最多奖励3分/级，机身抬升新进展最多1分/级；
都按该级高度归一化、只计历史新高，回落后再伸腿不重复得分。真实支撑进入收腿阶段奖励0.5分/级。
失败扣2分，靠近台阶尝试期间扣1分/s；所有事件/位移项已抵消 RewardManager 的dt缩放。
爬阶vx正奖励乘实际前进速度/目标速度（截断到0～1），避免低速命令下原地不动也拿高分。
爬阶yaw奖励乘0.1；姿态奖励允许30度pitch死区、严格约束roll，最高为平地该项的0.1。
垂直速度惩罚乘0.1，防止与向上运动冲突；其他地形保留原跟踪行为。

上行训练更新：`pyramid_stairs_inv` 上行楼梯区域只采样向前 0.15～0.30 m/s、yaw=0；
采用阶段高度辅助：前方0.15～0.65m扫描到2～22cm抬升时进入阶段1，目标高度0.40m；
底盘有向上支撑力且接近记录的台阶顶面持续0.08s后进入阶段2，目标降至0.30m，鼓励收腿。
两轮底部接近目标台阶高度、有接触且机身稳定持续0.2s，恢复原高度并给成功奖励。
爬阶期间及成功后0.5s允许底盘接触，上行区其余底盘接触持续0.2s才reset。
接近与操作分别计时：总尝试上限12s，靠近阶沿操作累计上限8s；靠近至少2s后连续3s无新进展判失败。
短暂后撤可用于恢复，不再因倒车命令立即reset。爬阶pitch上限60度、roll约40度，持续0.2s才reset；
严重翻倒（约75度）立即reset，腿长越界仍然终止。成功判定要求恢复稳定姿态。
状态机在当前物理步的终止/奖励计算前更新一次，避免支撑发生后仍按上一帧状态reset。
HUD ascent phase：0正常、1伸腿、2支撑收腿。支撑判断为力和高度近似，不是接触面分类；
没有直接写关节轨迹，需继续训练验证动作效果。策略观测与动作维度不变。
其他地形仍保留双向速度指令与底盘接触限制。新动作需继续训练；Play 的人工键盘仍可发送倒车指令。

任务 MyRobot-Velocity-Rough-Scan-v0 / MyRobot-Velocity-Rough-Scan-Play-v0
使用本机 Isaac Lab 的 ROUGH_TERRAINS_CFG，并调整为15%正向楼梯、40%反向楼梯（中心低、向外上行）、
10%随机方块、5%随机粗糙地面、5%正坡、5%反坡、20%平地。保留官方10行20列；
中间踏步宽0.8m、仅两级。上行区初始等级0，在第一级前约0.55～0.65m处正向站立起步；
连续完整通过两级3次才升一级（起点逐渐拉远，最高等级回到中心），连续失败2次降低一级。
这套课程改变起点距离，不降低150/200mm台阶高度；其他地形仍用官方行进距离课程。
该版本已加入平地列，用于保持直行与站立能力。地图配方有变化，旧 checkpoint 的 play 也会加载新地图。

保留本机轮腿动作、94维观测（49维原有输入+45维前后扫描）与地形感知姿态奖励；
并非照搬ANYmal的关节动作或187点扫描。初期vx±0.5、yaw±1，轮速投影仍生效。

训练重点看 `Metrics/base_height/first_step_pass_rate`、`two_step_pass_rate`、
`ascent_attempt_rate`、`ascent_stall_timeout_rate`，以及 `ascent_episode_count`（这些比例仅针对本批结束的上行回合）。
总reward上升不能替代通过率；单元测试只验证计分与状态逻辑，实际通过能力需重新训练和Play检查。

参考仓库核对：`state_machines/stair.py` 提供高度进展、轮子越阶和成功奖励，但
`pretrained/26_infantry/rough_rotation_stair/2026-07-30_09-16-27/params/env.yaml`
中 `enable_state_machines=false`、`stair_state_machine_cfg.enabled=false`。
该模型实际使用速度/高度/姿态、劈叉、接触与终止等奖励，并启用连续20步终止判断。
所以这里只借鉴其阶段反馈和持续终止设计，不声称这些专项奖励就是参考checkpoint成功的原因。

从已有94维Rough Scan模型续训本版：

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Rough-Scan-v0 \
  --num_envs 1024 \
  --resume \
  --checkpoint "/home/whc/桌面/RL/logs/rsl_rl/uz05_rough_scan/2026-09-21_18-09-29_ascent_reward_v4/model_11350.pt" \
  --reset_noise_std 0.10 \
  --max_iterations 3000 \
  --run_name two_step_150_200_dense_v2 \
  --headless
```

以下为从平地模型迁移的备选命令：

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Rough-Scan-v0 \
  --num_envs 1024 \
  --transfer_from /home/whc/桌面/RL/logs/rsl_rl/uz05_flat/2026-09-20_12-35-39_twist_workspace_v8/model_20294.pt \
  --reset_noise_std 0.10 \
  --max_iterations 3000 \
  --run_name official_rough_v1 \
  --headless
```

训练过程中周期性录制视频：`--video_interval` 按 PPO iteration 计算，`--video_length` 按仿真步数计算。下面命令每 500 轮录制一个 500 步视频，文件保存在本次 run 的 `videos/train/`：

```bash
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Rough-Scan-v0 \
  --num_envs 1024 \
  --transfer_from "/home/whc/桌面/RL/logs/rsl_rl/uz05_flat/2026-09-20_12-35-39_twist_workspace_v8/model_20294.pt" \
  --reset_noise_std 0.10 \
  --max_iterations 3000 \
  --run_name ascent_from_flat_v1_video \
  --video \
  --video_interval 500 \
  --video_length 500 \
  --headless
```

启用 `--video` 后会自动打开 camera 渲染，即使使用 `--headless` 也能录制；训练速度会下降，视频通常位于：

```text
logs/rsl_rl/uz05_rough_scan/<本次run>/videos/train/
```

默认训练 checkpoint 仍按 `save_interval=50` 保存。录像触发在第 500、1000、1500……个 PPO iteration 附近，不会改变 policy 的动作或观测。

训练后play使用 MyRobot-Velocity-Rough-Scan-Play-v0，--checkpoint指向94维新模型。
可选 --terrain_column 和 --terrain_level（0～9）；新配置列0～2正向楼梯，3～10反向楼梯，
11～12方块，13粗糙地面，14正坡，15反坡，16～19平地。play默认最低级并冻结课程。
实际高度在每行的难度区间内变化。原49维checkpoint不能直接用于此任务play。

## 备选：带45维扫描的单级台阶课程

训练与play指令、迁移规则、课程判据和sim2sim观测定义见 [step_course.md](step_course.md)。
新任务为 MyRobot-Velocity-Step-Course-v0，首次从平地模型使用 --transfer_from，
不是 --resume；课程模型观测94维，不能用旧49维Steps任务加载。

## 历史任务：150 / 200 mm 台阶与楼梯

新增 MyRobot-Velocity-Steps-v0 / MyRobot-Velocity-Steps-Play-v0，平地任务保持原设置。
训练地图：20% 平地、20% 起伏；150/200 mm 单级上台阶、单级下台阶、
150 mm 连续下楼梯、200 mm 连续上楼梯各 10%。单级台阶由中心平台/凹台形成，
楼梯是四周环绕的方形阶梯，踏步深 0.40 m、中心出生平台宽 2 m。
高度固定为 0.15 / 0.20 m，不从小台阶递增；地图生成按列分型，关闭等级晋升。
波浪参数 amplitude_range 为 0.01～0.03 m（不是台阶高度）。

初期指令 vx ±0.5 m/s、yaw ±1 rad/s，高度目标 0.30～0.40 m。
局部 0.2×0.2 m 射线网格的有限命中高度中位数作为地面参考，统一用于高度
观测、奖励和指标；边缘处仍可能跳变，因此放宽高度惩罚并关闭直行 pitch 辅助项。
完整地形扫描不进入 policy，观测维度不变，可加载平地 checkpoint；没有提前感知
台阶的观测，能否学会 200 mm 越障需实际训练验证。

先查看地形（零动作只验证场景，机器人可能倒下）：

```bash
conda activate env_isaaclab
python Wheel-Leg/scripts/debug_env.py --task MyRobot-Velocity-Steps-v0 --num_envs 10 --steps 1000
```

从平地最终 checkpoint 续训，日志写入 logs/rsl_rl/uz05_steps：

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Steps-v0 \
  --num_envs 1024 \
  --resume \
  --checkpoint /home/whc/桌面/RL/logs/rsl_rl/uz05_flat/2026-09-20_12-35-39_twist_workspace_v8/model_20294.pt \
  --reset_noise_std 0.10 \
  --max_iterations 3000 \
  --run_name steps_150_200_v1 \
  --headless
```

台阶 play 使用 --task MyRobot-Velocity-Steps-Play-v0，其余参数沿用 play 命令。
此任务使用与训练相同的完整地图；单环境默认从平地开始，可用 --terrain_column
选择出生列：0/1 平地，2 下台阶150mm，3 下台阶200mm，4 上台阶150mm，
5 上台阶200mm，6 连续下楼梯150mm，7 连续上楼梯200mm，8/9 起伏。
台阶测试初期请设 |vx|≤0.5、|yaw|≤1；可以用 `--command_vx`、`--command_yaw`
和 `--command_height` 固定测试指令。

使用现有平地模型预览 150 mm 上台阶（未训练越障能力）：

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/play.py \
  --task MyRobot-Velocity-Steps-Play-v0 \
  --num_envs 1 \
  --terrain_column 4 \
  --checkpoint /home/whc/桌面/RL/logs/rsl_rl/uz05_flat/2026-09-20_12-35-39_twist_workspace_v8/model_20294.pt \
  --real_time
```

Play 默认开启 Isaac Sim 窗口内交互：W/S 前后移动，A/D 转向，↑/↓ 调整前进速度，←/→ 调整转向速度，I/K 调整目标高度，H 恢复默认高度，Space 急停，R 重置环境。速度和 yaw 默认使用平滑过渡，并在送入 policy 前执行与训练一致的轮速工作空间投影。可用 `--no_keyboard`、`--no_hud` 或 `--no_follow_camera` 关闭对应功能。

### 交互 Play 操作说明

Play 使用 Isaac Sim 窗口中的键盘输入，不启动网页服务器，也不直接控制关节；所有键盘输入都会先更新 velocity/base-height command，再由训练好的 policy 输出轮腿动作。

| 按键 | 操作 |
|---|---|
| `W` | 按住前进，松开后目标 `vx` 回到 0 |
| `S` | 按住后退，松开后目标 `vx` 回到 0 |
| `A` | 按住左转，松开后目标 yaw rate 回到 0 |
| `D` | 按住右转，松开后目标 yaw rate 回到 0 |
| `↑` / `↓` | 增大 / 减小前进速度档位 `v_set`，范围 0.1～1.0 m/s |
| `←` / `→` | 减小 / 增大转向速度档位 `w_set`，范围 0.1～5.0 rad/s |
| `I` / `K` | 增大 / 减小目标机身高度，每次 5 mm |
| `H` | 恢复启动时的默认机身高度 |
| `Space` | 急停，将当前 `vx`、`vy`、yaw rate 立即置零；速度档位和高度设定保留 |
| `R` | Reset 当前仿真环境和机器人状态 |

W/S 和 A/D 是“按住有效、松开归零”，不会累加速度。速度和 yaw 默认使用 slew-rate 平滑，避免键盘按下或松开造成 command 突变；使用 `--no_command_smoothing` 可以关闭平滑。

### HUD 和相机显示

默认会打开 Isaac Sim 内的 `UZ-05 Play` HUD，并显示：

- Command：目标 vx、目标 yaw rate、轮速工作空间投影后的有效 vx/yaw、目标高度、`v_set` 和 `w_set`；
- Robot：实际 vx、vy、yaw rate、roll、pitch 和实际机身高度；
- Terrain：当前 terrain type index 和 terrain level；
- 底部操作提示。

默认使用自由镜头，不自动跟随机器人。默认开启 episode 终止条件及自动 reset（包括超时、倒地），也可按 R 手动重置。可用 `--follow_camera` 开启跟随，`--no_auto_reset` 关闭自动重置。R 在与 policy 相同的 inference_mode 内执行环境和策略状态重置。

HUD 的 `height local` 使用训练相同的 chassis 和局部地面参考，可与 `height target` 比较；`height error` 为实际减目标。`ground Z` 与 `base world Z` 是世界坐标，不应直接拿世界 Z 和目标高度比较。台阶边缘的局部地面中位数可能跳变。

Play 初始 `v_set` 为 0.3 m/s（不超过任务速度范围），高度键限幅采用当前任务的训练高度范围。命令行固定值仍优先于键盘设定，Space 可将固定速度指令也置零。可用以下参数关闭功能：

```bash
--no_keyboard             # 关闭键盘控制
--no_hud                  # 关闭 HUD
--no_follow_camera        # 关闭跟随相机
--no_command_smoothing    # 关闭速度/yaw 平滑
```

也可以使用命令行固定指令进行重复测试：

```bash
python Wheel-Leg/scripts/rsl_rl/play.py \
  --task MyRobot-Velocity-Steps-Play-v0 \
  --num_envs 1 \
  --terrain_column 4 \
  --checkpoint /absolute/path/to/model.pt \
  --command_vx 0.30 \
  --command_yaw 0.0 \
  --command_height 0.32 \
  --no_keyboard \
  --real_time
```

## 历史任务：按轮速工作空间压缩 vx/yaw

`--command_vx`/`--command_yaw` 会在策略观测计算前调用与训练相同的轮速投影，
因此命令行测试和训练使用一致的有效速度。

期望指令范围为 vx ±1 m/s、yaw ±5 rad/s。每次采样后按照差速轮关系计算：
`wheel_left=(vx-0.210335*yaw)/0.055`、`wheel_right=(vx+0.210335*yaw)/0.055`。
若任一轮超过策略轮速工作空间 20 rad/s，则同时将 vx 和 yaw 乘同一个比例，
保持原本的运动方向，不再分别裁剪两个指令。纯 yaw=5 rad/s 约需 19.13 rad/s
轮速，可以保留；vx=1、yaw=5 的组合会被整体压缩。混合转弯不再使用固定 ±1
的 yaw 上限。轮速 20 rad/s 是当前动作 scale，不是电机 60 rad/s 的硬件上限，
后续若高速转弯仍不稳，可把 max_wheel_speed 降至 18 留出动态余量。

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Flat-v0 \
  --num_envs 1024 \
  --resume \
  --load_run 2026-09-20_10-20-28_straight_pitch_range_v7 \
  --checkpoint model_17295.pt \
  --reset_noise_std 0.10 \
  --max_iterations 3000 \
  --run_name twist_workspace_v8 \
  --headless
```

## 历史任务：扩大高度/转向范围，纠正持续直行 pitch

高度范围 0.25～0.40 m；原地转向 yaw ±2 rad/s，混合行走暂保留 ±1 rad/s。
vx 仍为 ±1 m/s，前后采样各 50%。保留高度专项、20% 静止采样与防漂移奖励。
新增 straight_pitch：仅非零 vx、零 yaw 时生效，速度和高度指令每次重采样后
均等待至少 1 秒；允许约 3° pitch，超出部分按 0.1 rad 归一化平方惩罚，
权重 -0.5，未加权上限 4。时间门控不代表已经达到物理稳态，需 play 验证。
高度扩大后，低/中/高指标的分界随范围变化，不直接对比旧分组数值。

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Flat-v0 \
  --num_envs 1024 \
  --resume \
  --load_run 2026-09-19_18-49-10_height_stand_bidirectional_v6 \
  --checkpoint model_14296.pt \
  --reset_noise_std 0.10 \
  --max_iterations 3000 \
  --run_name straight_pitch_range_v7 \
  --headless
```

## 历史任务：动态高度跟踪 + 前后均衡 + 静止防漂

高度范围保持 0.28～0.34 m、yaw 保持 ±1 rad/s。高度指数奖励权重 3、核系数 1000，
高度 L2 权重 -1；保留上一轮 vx 强化。20% 速度指令为静止，剩余指令按
直行/转向/混合 = 50%/30%/20% 采样。高度每 4～8 秒重采样，70% 概率选择
与上次目标位于中点另一侧的端点，30% 均匀随机；回合初始化时随机选择端点。
这两种指令独立采样，静止指令不保证覆盖完整的升降过程。

非零 vx 指令的前进/后退概率改为 50%/50%。新增 stand_still_drift：仅当
vx、vy、yaw 指令全为零（容差 1e-6）时，惩罚水平速度 / 0.05 m/s 的逐轴
Huber 损失（beta=1，权重 -0.2，上限 10）；原有静止速度 L2 保留。
不惩罚竖直升降或关节平衡动作，也不增加位置保持目标。
新增 stand_vx_bias_mps（正值为向前偏）、stand_speed_xy_mps、stand_sample_count，
以及 stand_path_length_m（每个含静止样本回合的静止段速度积分路程，不是净位移；
包含制动瞬态，受静止持续时间影响）。sample_count 为 0 时对应指标无效。

新增 height_low/mid/high_actual_cm、target_cm、mae_cm 和 sample_count；
高度范围等分三档，计数为 0 的分组指标无效。检查低、高目标下实际高度是否分开，
不要仅看总体 MAE。play 已在覆盖指令后重新计算观测。

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Flat-v0 \
  --num_envs 1024 \
  --resume \
  --load_run 2026-09-19_14-08-37_mass_fixed_resume_8298 \
  --checkpoint model_11297.pt \
  --reset_noise_std 0.10 \
  --max_iterations 3000 \
  --run_name height_stand_bidirectional_v6 \
  --headless
```

## 历史任务：质量修正后加强 vx 跟踪

保持质量、动作映射、指令分布、yaw 和高度奖励不变；vx Gaussian 权重 3 → 4，
vx Huber 权重 -0.6 → -1.2，核宽度不变。总 reward 与旧版本不宜直接比较。
新增直行前进/后退、低速/高速 vx MAE；速度分界为 |cmd_vx| = 0.5 m/s，
这些分组包含指令切换瞬态，各自配套 sample_count，计数为 0 时 MAE 无效。

从质量修正后的 checkpoint 继续，适度恢复探索（不更改观测或动作维度）：

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Flat-v0 \
  --num_envs 1024 \
  --resume \
  --load_run 2026-09-19_14-08-37_mass_fixed_resume_8298 \
  --checkpoint model_11297.pt \
  --reset_noise_std 0.10 \
  --max_iterations 3000 \
  --run_name mass_fixed_vx_v4 \
  --headless
```

主要比较最近 100 轮的 vx_mae_straight_mps、vx_mae_mixed_mps；同时确认
height_mae_cm、yaw_mae_radps 和 time_out 没有明显退化。训练过程中先观察趋势，
最终用固定指令 play 检查稳态误差与切换后的响应，不能仅凭总回报判断改善。

## 历史任务：加强直行与动态高度跟踪

直行/原地转向/混合指令比例为 50%/30%/20%，关闭额外的强制反向采样。
普通随机采样仍可能改变 vx 正负号；速度范围保持 ±1 m/s、±1 rad/s。
目标高度为 0.28～0.34 m，每 4～8 秒重采样。reward 权重沿用上一版。

从已确认存在的最终 checkpoint 续训，并恢复探索标准差到 0.15：

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Flat-v0 \
  --num_envs 1024 \
  --resume \
  --load_run 2026-09-18_19-15-25 \
  --checkpoint model_5299.pt \
  --reset_noise_std 0.15 \
  --run_name straight_height_v3 \
  --headless
```

新增诊断指标：

- `vx_mae_straight_mps`、`vx_mae_turn_mps`、`vx_mae_mixed_mps`：各模式下的 vx MAE。
  每次 reset 按该批环境的对应模式样本汇总；配套的 `*_sample_count` 为 0 时，该 MAE 无效。
- `leg_action_at_limit_fraction`、`wheel_action_at_limit_fraction`：分别记录腿、轮动作达到限幅的比例，不等于力矩饱和率。
- `height_actual_cm`、`height_target_cm`：回合内实际/目标高度均值。
- `height_bias_cm`：实际高度减目标高度的均值，负值表示偏低；与 `height_mae_cm` 一起观察，避免正负误差相互抵消。

这里的高度指标是均值，判断升降滞后仍需查看 play 时的高度时间序列。

## 之前的命令记录

查看已有环境
conda env list

conda activate env_isaaclab 

退出
conda deactivate

tensorboard --logdir logs/rsl_rl/wheel_leg --host 0.0.0.0 --port 6006

零动作 step 模式。
python Wheel-Leg/scripts/preview_initial_state.py --task WheelLeg-v0 --num_envs 1 --zero_action_step --real_time

普通第一瞬间预览还是：
python Wheel-Leg/scripts/preview_initial_state.py --task WheelLeg-v0 --num_envs 1

reset 后静态预览是：
python Wheel-Leg/scripts/preview_initial_state.py --task WheelLeg-v0 --num_envs 1 --reset



python scripts/rsl_rl/train.py   --task WheelLeg-v0   --num_envs 1024   --resume   --load_run 2026-08-21_17-49-04   --checkpoint model_2850.pt

/home/whc/桌面/RL/Wheel-Leg/logs/rsl_rl/wheel_leg/2026-08-22_23-44-05/model_5500.pt

python scripts/rsl_rl/train.py   --task WheelLeg-v0   --num_envs 1024   --resume   --load_run 2026-08-23_16-16-00 --checkpoint model_7550.pt 

cd /home/whc/桌面/RL/Wheel-Leg
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

python scripts/rsl_rl/play.py \
  --task MyRobot-Velocity-Flat-Play-v0 \
  --num_envs 1 \
  --checkpoint /home/whc/桌面/RL/Wheel-Leg/logs/rsl_rl/uz05_flat/2026-09-06_18-03-48/model_6798.pt \
  --real_time

  /home/whc/桌面/RL/Wheel-Leg/logs/rsl_rl/uz05_flat/2026-09-05_19-30-51/model_3799.pt


平地
/home/whc/桌面/RL/logs/rsl_rl/uz05_flat/2026-09-20_12-35-39_twist_workspace_v8/model_20294.pt
