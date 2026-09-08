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
  --web_port 8768 \
  --web_frame_fps 10 \
  --web_frame_quality 92 \
  --web_frame_width 1920 \
  --web_frame_height 1080 \
  --real_time

  http://127.0.0.1:8768

  /home/whc/桌面/RL/Wheel-Leg/logs/rsl_rl/uz05_flat/2026-09-05_19-30-51/model_3799.pt