"""Stratified gentle-terrain evaluation; no training or automatic promotion."""
import json
import math
from pathlib import Path


def wilson_lower(successes, total, z=1.96):
    if total == 0:
        return 0.0
    p = successes / total
    return (p + z*z/(2*total) - z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


def run_terrain_evaluation(env, policy, app, force_velocity, args, checkpoint, output):
    import torch
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply
    from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations import base_height

    raw = env.unwrapped
    terrain, robot = raw.scene.terrain, raw.scene['robot']
    if not getattr(raw.cfg, 'terrain_relative_height', False):
        raise ValueError('--eval_terrain requires MyRobot-Velocity-Gentle-Play-v0')
    cfg = terrain.cfg.terrain_generator
    names = list(cfg.sub_terrains)
    weights = [cfg.sub_terrains[n].proportion for n in names]
    cumulative, value = [], 0.
    for weight in weights:
        value += weight / sum(weights)
        cumulative.append(value)
    columns = {name: [] for name in names}
    for col in range(cfg.num_cols):
        index = next(i for i, bound in enumerate(cumulative) if col/cfg.num_cols+.001 < bound)
        columns[names[index]].append(col)
    height_term = raw.command_manager.get_term('base_height')
    velocity_term = raw.command_manager.get_term('base_velocity')
    body_ids, _ = robot.find_bodies(height_term.cfg.body_name)
    body_cfg = SceneEntityCfg('robot', body_ids=body_ids)
    cases = [('forward', .3, 0.), ('reverse', -.3, 0.),
             ('turn_positive', 0., .5), ('turn_negative', 0., -.5), ('stand', 0., 0.),
             ('mixed_positive', .3, .5), ('mixed_negative', .3, -.5)]
    settle = math.ceil(args.eval_settle_s/raw.step_dt)
    steps = math.ceil(args.eval_measure_s/raw.step_dt)
    trials, groups = [], []
    interrupted = False
    for name, cols in columns.items():
        for case, vx, yaw in cases:
            results = []
            for trial in range(args.terrain_eval_trials):
                if not app.is_running():
                    interrupted = True
                    break
                row, col = trial % cfg.num_rows, cols[(trial // cfg.num_rows) % len(cols)]
                terrain.terrain_levels[0], terrain.terrain_types[0] = row, col
                terrain.env_origins[0] = terrain.terrain_origins[row, col]
                raw.scene.env_origins[0] = terrain.env_origins[0]
                env.reset()
                if hasattr(policy, 'reset'):
                    policy.reset(torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device))
                values = []
                signed_travel = 0.
                approach = math.ceil(2./raw.step_dt) if name.startswith('slope') and vx == 0. else 0
                approach_start = robot.data.root_pos_w[0,:2].clone()
                approach_ok = not approach
                status = 'complete'
                start = direction = end = None
                for step in range(approach + settle + steps):
                    if not app.is_running():
                        status, interrupted = 'interrupted', True
                        break
                    velocity_term.is_standing_env[:] = False
                    velocity_term.time_left[:] = float('inf')
                    height_term.time_left[:] = float('inf')
                    force_velocity(env, .3 if step < approach else vx, 0. if step < approach else yaw)
                    height_term.command[:, 0] = .30
                    if approach and step == approach:
                        approach_ok = float((robot.data.root_pos_w[0,:2]-approach_start).norm()) >= .4
                    if step == approach + settle:
                        start = robot.data.root_pos_w[0, :2].clone()
                        direction = quat_apply(robot.data.root_quat_w[0:1],
                            torch.tensor([[1., 0., 0.]], device=raw.device))[0, :2]
                        direction = direction / direction.norm().clamp_min(1.e-6)
                    _, _, done, _ = env.step(policy(env.get_observations()))
                    if bool(done[0]):
                        status = 'terminated'
                        break
                    if step >= approach + settle:
                        signed_travel += float(robot.data.root_lin_vel_b[0,0]) * raw.step_dt * (1 if vx >= 0 else -1)
                        gravity = robot.data.projected_gravity_b[0]
                        roll = torch.atan2(gravity[1], -gravity[2]).abs()
                        pitch = torch.atan2(gravity[0], gravity[1:].norm()).abs()
                        values.append([float((robot.data.root_lin_vel_b[0,0]-vx).abs()),
                                       float((robot.data.root_ang_vel_b[0,2]-yaw).abs()),
                                       float((base_height(raw, body_cfg)[0,0]-.30).abs()),
                                       float(torch.maximum(roll, pitch)),
                                       float(robot.data.root_lin_vel_b[0,:2].norm())])
                        end = robot.data.root_pos_w[0,:2].clone()
                means = [sum(v[i] for v in values)/len(values) for i in range(5)] if values else None
                progress = float(torch.dot(end-start, direction)) * (1 if vx >= 0 else -1) if end is not None else 0.
                drift = float((end-start).norm()) if end is not None else 0.
                finite = means is not None and all(math.isfinite(v) for v in means)
                posture_fraction = sum(v[3] <= math.radians(10) for v in values)/len(values) if values else 0.
                success = (status == 'complete' and approach_ok and len(values) == steps and finite
                           and means[0] <= .10 and means[1] <= .15 and means[2] <= .02
                           and posture_fraction >= .95)
                if vx:
                    success = success and (signed_travel if yaw else progress) >= .7*abs(vx)*steps*raw.step_dt
                else:
                    success = success and drift <= .30 and means is not None and means[4] <= .08
                result = dict(terrain=name, case=case, trial=trial, row=row, col=col,
                              status=status, success=bool(success), samples=len(values), approach_ok=approach_ok,
                              vx_mae=means[0] if finite else None, yaw_mae=means[1] if finite else None,
                              height_mae_m=means[2] if finite else None,
                              posture_in_10deg_fraction=posture_fraction,
                              progress_m=progress if math.isfinite(progress) else None,
                              drift_m=drift if math.isfinite(drift) else None)
                trials.append(result)
                results.append(result)
                # Incremental save survives a later simulator exit.
                with (Path(output)/'trials.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(result, allow_nan=False)+'\n')
                print(f'[TERRAIN EVAL] {name}/{case} {trial+1}/{args.terrain_eval_trials}: success={success}', flush=True)
                if interrupted:
                    break
            count = sum(r['success'] for r in results)
            groups.append(dict(terrain=name, case=case, episodes=len(results), successes=count,
                               success_rate=count/len(results) if results else 0.,
                               wilson_lower_95=wilson_lower(count, len(results))))
            if interrupted:
                break
        if interrupted:
            break
    # Conservative criterion per terrain AND command; never hide a weak direction.
    passed = not interrupted and all(g['episodes'] == args.terrain_eval_trials and
                                    g['wilson_lower_95'] >= .80 for g in groups)
    report = dict(checkpoint=str(checkpoint), seed=raw.cfg.seed, terrain_seed=cfg.seed,
                  stage=raw.cfg.terrain_stage, groups=groups, batch_passed=passed,
                  interrupted=interrupted,
                  note='A batch is not promotion. Require three held-out terrain seeds; '
                       'also run full-range flat regressions. Resets share startup dynamics. '
                       'Slope standing/turning require a 0.4 m approach before settling. '
                       'Success measures local traversal progress, not crossing the entire 12 m tile.')
    (Path(output)/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'[TERRAIN EVAL] batch_passed={passed}; report: {Path(output)/"report.json"}')
