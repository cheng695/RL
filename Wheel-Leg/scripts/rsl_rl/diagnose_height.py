"""Evaluate a checkpoint at fixed heights; save all samples and upright-only summaries."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/my_robot_lab"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--verify-inertials", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import my_robot_lab
from my_robot_lab.assets.robots.my_robot_ import LEG_JOINTS, WHEEL_BODY_NAMES, BASE_BODY_NAME

enable_extension("isaacsim.asset.importer.mjcf")

def main():
    task = "MyRobot-Velocity-Flat-Play-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent = handle_deprecated_rsl_rl_cfg(agent, metadata.version("rsl-rl-lib"))
    cfg.scene.num_envs = 12
    cfg.seed = 42
    cfg.episode_length_s = args.seconds + 5
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    cfg.commands.base_height.resampling_time_range = (1e9, 1e9)
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
    raw = env.unwrapped
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=raw.device)
    robot = raw.scene["robot"]
    masses = robot.root_physx_view.get_masses()[0].cpu().tolist()
    mass_report = dict(zip(robot.body_names, masses))
    print("IMPORTED_MASSES " + json.dumps(mass_report), flush=True)
    if args.verify_inertials:
        import mujoco
        reference = mujoco.MjModel.from_xml_path(str(ROOT / "source/my_robot_lab/my_robot_lab/assets/robots/UZ05_MJCF_real_params/xmls/uz05.xml"))
        inertias = robot.root_physx_view.get_inertias()[0].cpu().numpy().reshape(-1, 3, 3)
        coms = robot.root_physx_view.get_coms()[0].cpu().numpy()
        for i, name in enumerate(robot.body_names):
            j = reference.body(name).id
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, reference.body_iquat[j])
            rotation = rotation.reshape(3, 3)
            expected = rotation @ np.diag(reference.body_inertia[j]) @ rotation.T
            np.testing.assert_allclose(masses[i], reference.body_mass[j], rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(coms[i, :3], reference.body_ipos[j], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(inertias[i], expected, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(sum(masses), reference.body_mass.sum(), atol=1e-5)
        print("INERTIAL_CHECK PASSED: mass, COM and full inertia tensor for every rigid body", flush=True)
    joints, names = robot.find_joints(LEG_JOINTS, preserve_order=True)
    base = robot.find_bodies(BASE_BODY_NAME)[0][0]
    sensor = raw.scene.sensors["contact_forces"]
    wheel_ids = sensor.find_bodies(WHEEL_BODY_NAMES, preserve_order=True)[0]
    assert len(wheel_ids) == 2
    other_ids = [i for i in range(len(sensor.body_names)) if i not in wheel_ids]
    heights = torch.tensor([.28, .31, .34], device=raw.device).repeat_interleave(4)
    xml = ET.parse(ROOT / "source/my_robot_lab/my_robot_lab/assets/robots/UZ05_MJCF_real_params/xmls/uz05.xml")
    sites = {s.attrib["name"]: (b.attrib["name"], [float(x) for x in s.attrib["pos"].split()])
             for b in xml.iter("body") for s in b.findall("site")}
    pairs = []
    for side in ("left", "right"):
        for a, b in (("site_link3_to_link4", "site_link4_to_link3"), ("site_link6_to_link2", "site_link2_to_link6")):
            pair = []
            for site in (side + "_" + a, side + "_" + b):
                body, pos = sites[site]
                pair.append((robot.find_bodies(body)[0][0], torch.tensor(pos, device=raw.device).repeat(12, 1)))
            pairs.append(pair)
    history = []
    streak = torch.zeros(12, device=raw.device)
    with torch.inference_mode():
        for step in range(int(args.seconds / raw.step_dt)):
            raw.command_manager.get_command("base_velocity")[:] = 0
            raw.command_manager.get_command("base_height")[:, 0] = heights
            # Refresh observations AFTER overriding commands (including after resets).
            obs = env.get_observations()
            action = policy(obs)
            _, _, dones, _ = env.step(action)
            if hasattr(policy, "reset"):
                policy.reset(dones)
            tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
            forces = sensor.data.net_forces_w.norm(dim=-1)
            wheel_contact = (forces[:, wheel_ids] > 1).all(dim=1)
            other_contact = (forces[:, other_ids] > 1).any(dim=1)
            stable = ((tilt < .15) & wheel_contact & ~other_contact & ~dones.bool()
                      & (raw.episode_length_buf * raw.step_dt > 2)
                      & (robot.data.root_lin_vel_b.norm(dim=1) < .15)
                      & (robot.data.root_ang_vel_b.norm(dim=1) < .3))
            streak = torch.where(stable, streak + 1, 0)
            closure = []
            for pair in pairs:
                points = [robot.data.body_pos_w[:, i] + quat_apply(robot.data.body_quat_w[:, i], p) for i, p in pair]
                closure.append((points[0] - points[1]).norm(dim=1))
            record = dict(height=robot.data.body_pos_w[:, base, 2], tilt=tilt, stable=stable,
                          balanced=streak * raw.step_dt >= 1, done=dones, wheel_contact=wheel_contact,
                          other_contact=other_contact, q=robot.data.joint_pos[:, joints],
                          q_target=robot.data.joint_pos_target[:, joints], dq=robot.data.joint_vel[:, joints],
                          torque_est=robot.data.applied_torque[:, joints], action=action[:, :4],
                          closure=torch.stack(closure, dim=1))
            history.append({k: v.detach().cpu().numpy().copy() for k, v in record.items()})
    arrays = {k: np.stack([r[k] for r in history]) for k in history[0]}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "samples.npz", **arrays, targets=heights.cpu().numpy(), dt=raw.step_dt)
    report = {"checkpoint": args.checkpoint, "joint_names": names, "nominal_mass_com": True,
              "imported_body_masses_kg": mass_report, "total_mass_kg": sum(masses),
              "criteria": "tilt<0.15 rad, speed<0.15 m/s, angular speed<0.3 rad/s, both wheels >1N, other bodies <=1N, episode age>2s; continuously valid for 1s", "results": []}
    for group, height in enumerate((.28, .31, .34)):
        sub = {k: v[:, group*4:(group+1)*4] for k, v in arrays.items()}
        mask = sub["balanced"].astype(bool)
        row = dict(target=height, balanced_samples=int(mask.sum()), balanced_fraction=float(mask.mean()),
                   resets=int(sub["done"].sum()), instantaneous_stable_fraction=float(sub["stable"].mean()))
        if mask.any():
            for key in ("height", "tilt", "q", "q_target", "dq", "torque_est", "action", "closure"):
                row[key+"_mean"] = sub[key][mask].mean(axis=0).tolist()
            row["height_mae_cm"] = float(np.abs(sub["height"][mask]-height).mean()*100)
            row["leg_action_limit_fraction"] = float((np.abs(sub["action"][mask]) >= .999999).mean())
        report["results"].append(row)
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print("HEIGHT_DIAGNOSTIC " + json.dumps(report), flush=True)
    env.close()

try:
    main()
finally:
    app.close()
