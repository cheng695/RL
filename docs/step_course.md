# 单级台阶课程与迁移

任务：`MyRobot-Velocity-Step-Course-v0` / `MyRobot-Velocity-Step-Course-Play-v0`。
旧 Flat、固定 150/200mm Steps 保留；94维课程模型不能用49维任务加载。

## 地形与成功判据

借用 Isaac Lab TerrainGenerator、RayCaster、CurriculumManager。地图8行20列，
0～5列平地，6～12列单级上台阶，13～19列单级下台阶。
各行高度严格为25、50、75、100、125、150、175、200mm，初始全在第0行。
本阶段专门训练单级台阶；连续楼梯保留在原Steps任务作为后续测试地形。

每个障碍回合在2m宽中心平台出生，随机前进/后退，保持0.15～0.5m/s直行指令。
双轮沿要求方向越过平台边缘至少0.15m、保持在目标通道，双轮中心接近目标
平台高度+55mm（容差40mm）、两轮都有接触力、姿态倾斜小于约20°且垂直速度
小于0.15m/s，持续0.5秒才算成功。碰撞/姿态/腿长失败覆盖成功。
成功结束回合，并提供一次奖励；不能用存活时间或停在边缘代替成功。

每个环境连续成功3回合升一级，连续失败/超时2回合降一级；最高级保持200mm。
30%平地保留原vx±1、yaw±5的指令分布与轮速投影，不参与台阶升级。
姿态与高度精度约束随扫描高度差连续放松，离开台阶边缘后恢复。

日志：`step_up/down_success_rate`、`failure_rate`、`timeout_rate`及各自`episode_count`。
零计数的分组比率无意义；日志窗口的平均比率不是严格全局加权成功率。
`Curriculum/terrain_levels`是含平地在内的平均等级，不能据此认定所有环境已达200mm。

## 感知接口（sim2sim必须一致）

原有49维顺序保持：base_lin_vel(3)、base_ang_vel(3)、projected_gravity(3)、
velocity_commands(3)、height_command(1)、base_height(1)、base_height_error(1)、
joint_pos(14)、joint_vel(14)、actions(6)。末尾追加terrain_scan(45)，总计94。

扫描随车身yaw旋转，竖直向下；x=-0.8至0.8、y=-0.4至0.4，间距0.2m。
顺序为x外循环、y内循环（Isaac GridPattern ordering="yx"）；各点地面世界高度
减去车身下方0.2×0.2m参考网格的有限射线命中中位数，单位m，裁剪[-0.5,0.5]。
未命中点置0。局部参考网格全部未命中时回退到出生地形高度。
base_height和height_error也使用这个局部参考；flat模型与课程的这两项语义
在平地相同，在台阶上不同。扫描周期0.02s，动作维度6保持不变。

当前是理想射线输入，尚未实现真实相机遮挡、噪声、延迟或MuJoCo扫描接口。

## 迁移与继续训练

首次使用 `--transfer_from`：Actor、Critic第一层复制旧49列并将新增45列置零，
其余参数严格复制；不加载旧优化器，迭代从0开始。要求原始49维观测顺序、
无观测归一化及现有RSL-RL MLP结构；不符合时明确报错。迁移不会自动验证越障能力。
以后继续同一个94维模型使用 `--resume --checkpoint ...`，不要再使用transfer。

```bash
cd /home/whc/桌面/RL
conda activate env_isaaclab
python Wheel-Leg/scripts/rsl_rl/train.py \
  --task MyRobot-Velocity-Step-Course-v0 \
  --num_envs 1024 \
  --transfer_from /home/whc/桌面/RL/logs/rsl_rl/uz05_flat/2026-09-20_12-35-39_twist_workspace_v8/model_20294.pt \
  --reset_noise_std 0.10 \
  --max_iterations 3000 \
  --run_name scan_step_course_v1 \
  --headless
```

课程模型play（把checkpoint替换为课程训练实际文件）：

```bash
python Wheel-Leg/scripts/rsl_rl/play.py \
  --task MyRobot-Velocity-Step-Course-Play-v0 \
  --num_envs 1 --terrain_column 6 --terrain_level 5 \
  --checkpoint /absolute/path/to/course/model.pt \
  --real_time
```

第5行150mm、第7行200mm；上台阶列6～12，下台阶列13～19。
play固定难度、不自动升降级；手动vx建议0.15～0.5，yaw=0。
