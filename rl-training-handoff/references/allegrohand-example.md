# AllegroHand reference run

This is a concrete example of the generic workflow, not a required schema for
other robots.

## Artifacts

- Run: `allegro_60mm_dextreme_reorientation`
- Checkpoint: `model_2999.pt`
- Event data: `events.out.tfevents.*`
- Local bundle: `reorientation_training_bundle.json`

The bundle contains 33 scalar tags, complete step/value series, and the task,
observation, action, reward, termination, DR, and PPO metadata used for the run.

## Viser wiring

The local preview entrypoint constructs the *same* Reorientation environment as
training, loads the checkpoint with `strict=True`, gets the inference policy,
then injects a Viser server into MJLab's viewer:

```python
server = viser.ViserServer(
    port=8087,
    label="Allegro Hand DeXtreme Reorientation",
)
viewer = ViserPlayViewer(
    wrapped_env,
    policy,
    viser_server=server,
)
viewer.run()
```

The page is a live browser client at `http://localhost:8087`; each viewer step
advances MuJoCo and streams the updated hand/cube scene over Viser's websocket.

## Important verification lesson

The training script called `make_reorientation_env_cfg(...)` without passing
`tilt_degrees=30`, so the actual run had zero palm tilt even though the desired
physical scenario called for 30 degrees. Always verify requested physical
parameters in the effective training call, not only in a separate preview or
design document.

## Useful scalar groups

- `Episode_Reward/*`: orientation, position, action, smoothing, success bonus;
- `Episode_Metrics/*`: consecutive successes, success rates, orientation error,
  time-to-success, drop rate, position deviation;
- `Loss/*`, `Policy/mean_std`: PPO convergence and exploration;
- `Train/*`, `Perf/*`: episode length, reward, collection/learning throughput.
