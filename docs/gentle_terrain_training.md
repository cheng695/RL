# 从平地策略迁移到轻度复杂地形

日期：2026-10-21。实现了配置、离地高度、分组评估；尚未启动 Isaac Sim 或训练验证。

## 训练阶段

新任务 `MyRobot-Velocity-Gentle-v0`，测试任务 `MyRobot-Velocity-Gentle-Play-v0`。
平地任务保留原命令范围。通过环境变量 `UZ05_TERRAIN_STAGE` 显式选阶段，默认 1。

| 阶段 | 地形 | vx / yaw 范围 | 高度命令 |
|---|---|---|---|
| 1 | 25% 平地，75% 平滑随机起伏，地形高度 ±5 mm | ±2.5 m/s / ±12 rad/s | 0.25～0.40 m |
| 2 | 平地、±10 mm 起伏、正坡、反坡各 25%；坡度约 1.5～3° | ±2.5 / ±12 | 同上 |
| 3 | 平地、±20 mm 起伏、正坡、反坡各 25%；坡度约 5～10° | ±2.5 / ±12 | 同上 |
| 4 | 平地、±30 mm 起伏、正坡、反坡各 25%；坡度约 8.7～17° | ±2.5 / ±12 | 同上 |

随机起伏每 0.2 m 采样并插值；±5 mm 是地面高度幅度，峰谷差可以达 10 mm，并非 5 mm 台阶。坡度通过 tan(角度) 转成高度场斜率。第四阶段仍不是台阶训练。坡度范围实际在线性斜率 tan(角度) 上插值，因此第四阶段下限约为 atan(tan(17°)/2) ≈8.7°。

命令模式继承 20% 静止、40% 直行、20% 原地旋转、20% 边走边转；正负速度均覆盖。四个阶段都使用完整范围，非零命令沿用平地的低速优先分段采样：vx 幅值 0.08～0.5、0.5～1.0、1.0～2.5 m/s 的概率依次为 60%、25%、15%；yaw 幅值 0.15～1、1～2.5、2.5～5、5～12 rad/s 的概率为 60%、25%、10%、5%。这些比例是相应非零分量采样时的概率；正负各 50%。混合命令仍会投影到轮速可行域，所以最终执行命令的分布可能改变。高度范围 0.25～0.40 m，每回合采样一次。阶段中不额外增加推力扰动。模型质量及 COM 随机化在续训时读取 checkpoint 的 params/env.yaml；显式环境变量仍优先。

保留 25% 平地环境并覆盖原速度范围以缓解遗忘，仍须单独用 Flat-Play 检查原有中高速性能。

## 高度与姿态

原代码的高度是世界 Z。现在 Gentle 使用机身下方 0.4 × 0.4 m、间距 0.1 m 的向下射线网格，取有效地面高度均值：

`height = chassis_world_z - mean(local_ground_world_z)`

高度观测、高度误差观测、奖励、训练高度指标、HUD、motion trace 和评估统一此定义。平地继续用原世界 Z。缺失全部射线时数值暂用地形原点回退，同时无效射线触发终止，不能靠缺失数据获得成功。

这是局部地面参考高度，不是精确轮地接触距离，也不是机身最低点净空。在台阶、悬空、沟槽阶段应重新设计参考地面，不能直接照搬本阶段均值。

姿态目标仍是相对重力水平，保留 orientation_tracking 和 roll/pitch 死区惩罚，以及原角度终止。保留现有小角度容许范围，避免将越障瞬间所需的姿态调整全部消除；目前不加大权重，不强制机身贴合坡面。机身及其他非轮子触地终止保留，包含原 10 步启动宽限期。

新增越出所属地形块边界的终止（距边缘 0.5 m），越界不计成功。未接入原生按行走距离升级的 curriculum，因为静止、原地转向和倒车不能用同一个位移判据升级。TerrainGenerator 的 curriculum=True 仅用于固定每列地形种类；训练中地形行不会自动升级。

## 观测与续训兼容

第一阶段没有增加观测维度：复用已有高度与高度误差两个通道，读取局部地面；因此平地 checkpoint 的网络形状兼容。平地上新高度定义与原定义一致。

当前是依靠本体感知的轻度地形迁移，未向策略提供前方高度图。下一步如训练明显台阶、需要提前抬腿，应增加前方高度扫描及处理后的动作目标状态；届时必须显式迁移网络输入层或重新训练，不能直接严格加载旧网络。改变观测维度也要处理 critic 和归一化状态。本阶段没有实现这种网络扩展。

PPO 初始学习率设为 3e-4，续训保留 optimizer 动量并重设学习率，之后仍由 adaptive schedule 调整。奖励权重和动作滤波沿用已学好平地的设置。

## 第一轮训练

从当前已学好平地的策略开始，先进行 1000 次迭代：

```bash
conda activate env_isaaclab
UZ05_TERRAIN_STAGE=1 UZ05_TERRAIN_SEED=42 \
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Gentle-v0 \
  --num_envs 1024 \
  --resume \
  --checkpoint "/home/whc/桌面/RL/logs/rsl_rl/uz05_flat/2026-10-20_22-50-36_yaw_tracking_continue/model_5899.pt" \
  --max_iterations 1000 \
  --run_name gentle_stage1 \
  --headless
```

如果后来已有更好的平地 checkpoint，请替换路径。新日志目录为 `logs/rsl_rl/uz05_gentle/`。1000 是追加迭代数，不是绝对停止编号。代码未自动启动此命令。

建议先用 `--num_envs 16 --max_iterations 1 --run_name gentle_smoke` 验证导入、射线、接触及高度信号，确认没有报错后再运行正式规模；这不是效果验证。首轮出现显存/内存问题时先检查新传感器和地形开销。

## 通过标准与评估数量

`--eval_terrain` 每种地形分别测试前进 +0.3、后退 -0.3 m/s，原地 yaw ±0.5 rad/s，静止，以及 vx +0.3/yaw ±0.5 的两种混合命令，共 7 组。高度固定 0.30 m。

默认 **每个地形 × 命令组 100 回合**：每种地形 700 回合，阶段 1 共 1400，阶段 2/3/4 共 2800。`--terrain_eval_trials` 可调；少量回合用于检查运行，正式门槛建议保持 100 或以上。这不是每次训练迭代都执行，可在一个训练阶段结束时运行。

每回合重新 reset，排除 2 秒稳定时间，再测量 8 秒。坡面上的原地转向和静止会先用 0.3 m/s 前进 2 秒，且必须离开起点至少 0.4 m，避免只在中心平台上通过。接近过程中摔倒同样算失败。

单回合成功须同时满足：

- 全程没有终止，采样完整。
- vx MAE ≤0.10 m/s，yaw MAE ≤0.15 rad/s，离地高度 MAE ≤0.02 m。
- 测量时间至少 95% 的样本同时满足 |roll|、|pitch| ≤10°。
- 直行：沿初始朝向的有符号位移达到命令预期距离的 70%，倒车方向反转计算。
- 混合行走：沿机身前向累计运动距离达到预期的 70%，允许有意转弯。
- 静止/原地旋转：净平移 ≤0.30 m，平均平面速度 ≤0.08 m/s，防止来回运动抵消位移作弊。

这里检查的是局部地形上的运动能力，不是走完整块 12 m 地形；台阶阶段应采用明确的入口/出口穿越成功标准。

分组报告除成功率外还计算 Wilson 95% 置信区间下界，所有组下界 ≥0.80 才标记本批通过。例如 40/50=80% 的下界约 67%，不够稳健；100 回合约需 88 次成功才能满足下界要求。阈值是工程起点，需要结合实际通过视频检查，不是统计保证。

```bash
UZ05_TERRAIN_STAGE=1 UZ05_TERRAIN_SEED=1042 \
python Wheel-Leg/scripts/rsl_rl/play.py \
  --task MyRobot-Velocity-Gentle-Play-v0 \
  --num_envs 1 \
  --checkpoint "/path/to/gentle_checkpoint.pt" \
  --seed 1042 \
  --eval_terrain --terrain_eval_trials 100 \
  --eval_output logs/terrain_eval/stage1_seed1042 \
  --headless
```

再以 2042、3042 分别替换两处 seed 和输出目录，对同一候选 checkpoint 做两批评估。地形 seed 必须独立于训练；单改 `--seed` 不会改变已显式指定的地形 seed。输出目录必须是新目录。

保存 `env.yaml`、`agent.yaml`、逐回合 `trials.jsonl` 和汇总 `report.json`。中断或失败的组不算通过。程序不自动切换训练阶段，检查三批 report、平地回归和视频之后再把 `UZ05_TERRAIN_STAGE` 改成 2。

同一批 reset 仍共享本次启动时的质量/COM 随机化，且多个回合重复使用有限地形块，因此不能把它们当成完全独立的动力学和地形样本。不同启动 seed 与地形 seed 可扩大覆盖；调参反复使用的验证 seed 最后还需要一组未参与选择的测试 seed。

正式升级前额外检查原有高速、三个高度档、倒车混合转向和横坡。当前固定命令评估没有覆盖全部高度与速度空间，三批通过不代表全工作域已经掌握。

## 当前验证范围

完成 Python 编译检查，以及 CPU 下局部高度/无效射线回退、置信区间门槛检查；尚未运行仿真、训练或正式评估。导入 API 参照本机 IsaacLab 源码。需要首轮仿真确认 MJCF 导出层级中的 ray caster 路径及实际碰撞表现。

## PLAY 指定单一地形

使用 Gentle-Play 任务，通过 `--terrain` 选择固定类型：

| 参数值 | 地形 |
|---|---|
| `mixed` | 当前阶段的混合地形，默认行为 |
| `flat` | 平地 |
| `rough` | 短距离粗糙起伏，随机高度采样间距 0.2 m |
| `undulating` | 较缓、较长的起伏，随机高度采样间距 0.8 m |
| `slope_up` | 正金字塔坡，中心高、四周低 |
| `slope_down` | 倒金字塔坡，中心低、四周高 |
| `stairs_up` / `stairs_down` | 上、下台阶预留选项，当前选择时提示未启用 |

两种随机起伏使用相同高度场生成器，区别为起伏的空间尺度。正、反坡是整块地形形状；出生在中心平台，正坡向外开走会下坡，反坡向外开走会上坡。它们不是沿机器人前方无限延伸的单向坡道。

```bash
python Wheel-Leg/scripts/rsl_rl/play.py \
  --task MyRobot-Velocity-Gentle-Play-v0 \
  --num_envs 1 \
  --checkpoint "/path/to/model.pt" \
  --terrain rough --terrain_amplitude 0.01 --terrain_seed 42
```

替换最后一行为以下参数即可改变地形：

```bash
--terrain flat
--terrain undulating --terrain_amplitude 0.02 --terrain_seed 42
--terrain slope_up --terrain_slope_deg 3
--terrain slope_down --terrain_slope_deg 3
```

`--terrain_amplitude` 单位是米、步长 0.0025 m；`--terrain_slope_deg` 为正的角度大小，允许 (0, 17] 度，不表示已有策略能通过对应坡度。不指定幅度时取当前阶段的幅度；不指定坡度时，阶段 1/2 为 3°，阶段 3 为 10°，阶段 4 为 17°。

单一类型生成一块 12 × 12 m 地形，同 seed、同几何参数重复运行可复现该地形；R 重置机器人不会重新生成地形。键盘仍控制机器人运动和高度。`--terrain_seed` 与机器人随机 `--seed` 独立。

`--eval_terrain` 保持评估整个阶段地形组合，不接受单一地形覆盖；评估地形种子仍通过 `UZ05_TERRAIN_SEED` 指定。


### 交互 PLAY 的箭头和命令范围

Gentle-Play 启用期望/实际速度箭头，键盘 vx 范围为 ±2.5 m/s、yaw 为 ±12 rad/s，高度为 0.25～0.40 m。训练使用相同完整范围，并按前述分段概率偏向低速。PLAY 的可输入范围不代表当前地形上已经训练或验证通过；建议从低速开始检查。`--eval_terrain` / `--eval_tracking` 运行固定命令评估时会关闭箭头。


### 交互 PLAY 的 reset 行为

默认关闭超时终止，保留非轮子触地、倾覆、腿部超限及地形边界/射线失败终止；这些失败仍会自动 reset，也可随时按 R 手动 reset。此行为同时适用于 Flat 和 Gentle 交互 PLAY。显式 `--auto_reset` 启用包括超时在内的全部终止；显式 `--no_auto_reset` 关闭全部自动终止，仅手动 reset。`--eval_tracking` / `--eval_terrain` 始终按正式评估规则保留完整终止，不受交互 reset 开关影响。


第四阶段通过 `UZ05_TERRAIN_STAGE=4` 选择，例如从第三阶段合格 checkpoint 续训。单独查看 17° 反坡可在 Gentle-Play 命令后指定 `--terrain slope_down --terrain_slope_deg 17`；查看 ±30 mm 起伏可指定 `--terrain rough --terrain_amplitude 0.03`。
