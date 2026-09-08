---
name: rl-training-handoff
description: Use when a completed or paused RL run lives on a remote server and needs checkpoint inspection, TensorBoard viewing, local Viser preview, or a reproducible analysis bundle.
---

# RL Training Handoff

Use this skill when a user wants to inspect, visualize, or hand off a completed
reinforcement-learning run from a remote machine. It applies to dexterous hands,
bipeds, wheel-legged robots, and other MuJoCo/MJLab/RSL-RL-style projects.

The default outcome is a local, reproducible handoff:

1. identify the completed run and the exact checkpoint;
2. identify the remote Python environment and GPU without guessing;
3. pull the checkpoint, TensorBoard event files, and relevant cache/config data;
4. expose TensorBoard through an SSH tunnel;
5. start the project's matching local policy preview, preferably Viser;
6. write one machine-readable JSON bundle containing metadata and full scalar time series.

Do not retrain, modify reward code, or delete remote files unless the user explicitly asks.
Do not reuse a preview entrypoint from another task until its Actor observation shape,
action parameterization, and environment configuration match the checkpoint.

## Phase 1: discover the run

Inspect the remote run directory read-only first. Find:

- the newest checkpoint (`model_*.pt`, `.pth`, `.ckpt`, or project equivalent);
- TensorBoard event files (`events.out.tfevents.*`);
- the run configuration, cache, normalization statistics, and model metadata;
- whether the training process is still running;
- the exact interpreter/conda/venv used for training;
- GPU name and `torch.cuda.is_available()` in that interpreter.

Treat the remote path as opaque: paths containing spaces must be quoted for the
remote shell. If `scp` reports `ambiguous target`, escape spaces in the remote
spec (`Dexterous\\ Hand`) or use SFTP. Never use a broad recursive delete or
`rsync --delete` for this workflow.

## Phase 2: pull artifacts

Create a local run-specific directory such as:

```text
logs/<experiment>/remote-<date>/
```

Pull at minimum:

- the selected checkpoint;
- all event files for the run (not only the last one if multiple exist);
- the exact cache used by reset, if it is not already local;
- task/config files needed to reconstruct the environment.

After transfer, compare SHA-256 hashes or byte sizes. Keep the original remote
paths in the handoff metadata.

## Phase 3: TensorBoard display

If TensorBoard is installed only on the server, start it bound to loopback on
the server:

```bash
<remote-python> -m tensorboard.main \
  --logdir "<remote-logdir>" \
  --host 127.0.0.1 --port 6006
```

Then create a local SSH tunnel:

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<host> -N
```

If local port 6006 is occupied, choose another local port and keep the remote
port unchanged. Verify with `curl -I http://127.0.0.1:<local-port>/` before
reporting the URL. The user-facing page is `http://localhost:<local-port>`.

Inspect at least these scalar groups when diagnosing learning:

- reward terms separately, not only mean reward;
- task success count/rate and time-to-success;
- orientation/pose/velocity errors;
- drop, timeout, invalid-state, and workspace termination rates;
- policy standard deviation/entropy;
- learning rate, value loss, surrogate loss, and throughput.

## Phase 4: local Viser preview

Use the project's existing play/viewer entrypoint first. The checkpoint must be
loaded with the same environment configuration used for training, including
robot asset, object size/mass, observation history, action scaling, controller
smoothing, normalization, and privileged/Oracle state choices.

For any simulator, the preview contract is:

1. load the checkpoint and construct the matching inference environment;
2. reset the environment and obtain observations;
3. loop `observation -> policy(observation) -> environment.step(action)`;
4. update Viser transforms/geometries from the simulator state every step;
5. keep the server alive, print its URL, and stop on dimension mismatch or an
   explicit user stop.

For MJLab projects, the standard implementation is:

```python
server = viser.ViserServer(port=8087, label="RL preview")
viewer = ViserPlayViewer(
    wrapped_env,
    runner.get_inference_policy(device=device),
    viser_server=server,
)
print("Viser preview: http://localhost:8087")
viewer.run()
```

`ViserPlayViewer` advances the wrapped MuJoCo/MJLab environment with the loaded
inference policy and streams the current scene to a Viser WebSocket/HTTP server.
The browser connects to the printed URL; it is not a static screenshot. For a
custom MuJoCo scene without `ViserPlayViewer`, use `viser.ViserServer`, create
the project's scene adapter, and update it every simulation step (in the
current reference implementation this is `scene.update_from_mjdata(data)`).

Never silently fall back to a zero policy and call that a policy preview. If
the checkpoint and environment dimensions disagree, stop and report the exact
dimensions instead of relaxing `strict=True`.

## Phase 5: create the analysis bundle

Run [collect_tensorboard_bundle.py](scripts/collect_tensorboard_bundle.py) in
an environment that has TensorBoard installed. Pass metadata as one JSON
object using these stable top-level keys (empty objects are acceptable when a
run does not expose a category):

```json
{
  "run": {}, "checkpoint": {}, "event_files": {}, "env": {},
  "obs": {}, "action": {}, "reward": {}, "training": {}, "preview": {}
}
```

The script validates these keys and stores the object unchanged alongside the
scalar data. It stores:

- run/checkpoint/event-file paths;
- task, observation, action, reward, termination, DR, and PPO metadata supplied
  by the caller;
- complete scalar series with `step`, `wall_time`, and `value`;
- first/last/min/max/count summaries for every scalar tag.

The output is one JSON file. Preserve the raw event file beside it when
possible; the JSON is an analysis handoff, not a replacement for raw data.

## Completion report

Report only evidence-backed facts:

- checkpoint selected and hash/size;
- TensorBoard URL and event tags found;
- Viser URL and the exact environment/policy dimensions loaded;
- bundle path;
- whether the preview is still running;
- blockers such as missing CUDA, missing TensorBoard, incompatible checkpoint,
  or an unverified task parameter (for example a requested palm tilt not passed
  into the training config).

## Reference

Read [allegrohand-example.md](references/allegrohand-example.md) only when
working on the current AllegroHand/MJLab example or when you need a concrete
reference for the Viser and TensorBoard wiring. The reference is illustrative;
do not assume hand-specific names or dimensions for biped and wheel-legged runs.
