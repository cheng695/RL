"""Check terrain mapping and contact exception only apply to ascent columns."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import pytest

path = Path(__file__).parents[1] / 'source/my_robot_lab/my_robot_lab/tasks/manager_based/locomotion/velocity/mdp/ascent.py'
tree = ast.parse(path.read_text())
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
scope = {'torch': torch,
         'illegal_contact_after_steps': lambda env, *args: torch.ones(env.num_envs, dtype=torch.bool),
         'contact_sensor_contact': lambda env, **kwargs: torch.ones(env.num_envs)}
exec(compile(tree, str(path), 'exec'), scope)


def test_ascent_contact_exception():
    cfg = NS(num_cols=20, sub_terrains={'pyramid_stairs': NS(proportion=.15),
             'pyramid_stairs_inv': NS(proportion=.4), 'flat': NS(proportion=.45)})
    env = NS(num_envs=20, device='cpu', scene=NS(terrain=NS(
        cfg=NS(terrain_generator=cfg), terrain_types=torch.arange(20))))
    expected = (torch.arange(20) >= 3) & (torch.arange(20) <= 10)
    term = NS(phase=expected.long(), cooldown=torch.zeros(20), contact_time=torch.zeros(20),
              _contact_check=-1, check_state=lambda: None)
    env.common_step_counter = 1
    env.step_dt = .02
    env.command_manager = NS(get_term=lambda name: term)
    assert torch.equal(scope['ascent_mask'](env), expected)
    assert torch.equal(scope['unsupported_base_contact'](env, 1, 100, None), ~expected)
    assert torch.equal(scope['unsupported_contact_cost'](env, .1, None), (~expected).float())
    term.phase.zero_()
    for step in range(2, 12):
        env.common_step_counter = step
        result = scope['unsupported_base_contact'](env, 1, 100, None)
    assert result.all()


@pytest.fixture
def ascent_fixture():
    class Parent:
        def _update_command(self):
            self._env.common_step_counter += 1
        def reset(self, env_ids=None):
            return {}
    parsed = ast.parse(path.read_text())
    parsed.body = [n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'AscentHeightCommand']
    namespace = {'torch': torch, 'UniformBaseHeightCommand': Parent,
                 'ascent_mask': lambda env: torch.tensor([True])}
    exec(compile(parsed, str(path), 'exec'), namespace)
    term = namespace['AscentHeightCommand'].__new__(namespace['AscentHeightCommand'])
    term.phase = torch.zeros(1, dtype=torch.long)
    term.completed_steps = torch.zeros_like(term.phase)
    term.course_success = torch.zeros(1)
    term.highest_completed_top = torch.full((1,), -torch.inf)
    term._state_check = term._contact_check = term._tilt_check = -1
    for name in ('near_time', 'stall_time', 'support_time', 'support_event', 'entry_ground',
                 'entry_chassis_z', 'best_wheel_lift', 'best_body_lift', 'wheel_lift', 'body_lift',
                 'contact_time', 'tilt_time', 'success_streak', 'failure_streak', 'attempts'):
        setattr(term, name, torch.zeros(1))
    for name in ('elapsed', 'settle', 'top', 'cooldown', 'success', 'failed', 'crossing', 'best_progress', 'progress', 'height_tolerance'):
        setattr(term, name, torch.zeros(1))
    term.entry_xy = torch.zeros(1, 2)
    term.direction = torch.zeros(1, 2)
    term.height_command = torch.tensor([[.35]])
    term.nominal = term.height_command.clone()
    term.cfg = NS(asset_name='robot', height_range=(.30, .40))
    term.wheels = term.wheel_contacts = [1, 2]
    term.chassis_contacts = term._height_body_ids = [0]
    data = NS(root_pos_w=torch.tensor([[0., 0., .35]]), root_quat_w=torch.tensor([[1., 0., 0., 0.]]),
              body_pos_w=torch.tensor([[[0., 0., .35], [0., .2, .055], [0., -.2, .055]]]),
              projected_gravity_b=torch.tensor([[0., 0., -1.]]))
    forces = torch.zeros(1, 3, 3)
    term._env = NS(step_dt=.02, common_step_counter=0, episode_length_buf=torch.ones(1), scene={'robot': NS(data=data),
        'terrain_scan': NS(data=NS(ray_hits_w=torch.tensor([[[.3, 0., .15]]]))),
        'contact_forces': NS(data=NS(net_forces_w=forces))},
        command_manager=NS(get_command=lambda name: torch.tensor([[.2, 0., 0.]])))
    term.reference_ground_height = lambda: torch.zeros(1)
    term._env.command_manager.get_term = lambda name: term
    return term, data, forces


def test_extend_support_retract_and_success(ascent_fixture):
    term, data, forces = ascent_fixture
    term._update_command()
    assert term.phase.item() == 1
    assert abs(term.height_command.item()-.4) < 1e-6
    data.root_pos_w[0, 0] = .1
    data.body_pos_w[0, 1:, 0] = .1
    term._update_command()
    assert term.progress.item() > .09
    data.root_pos_w[0, 0] = 0.
    data.body_pos_w[0, 1:, 0] = 0.
    term._update_command()
    assert term.progress.item() == 0
    data.root_pos_w[0, 0] = .1
    data.body_pos_w[0, 1:, 0] = .1
    term._update_command()
    assert term.progress.item() == 0  # revisiting old position earns nothing
    # Side impact cannot trigger retraction.
    forces[0, 0, 0] = 50.
    for _ in range(8):
        term._update_command()
    assert term.phase.item() == 1
    data.body_pos_w[0, 0, 2] = .28
    forces[0, 0, 2] = 10.
    for _ in range(4):
        term._update_command()
    assert term.phase.item() == 2
    assert term.support_event.item() == 1
    assert abs(term.height_command.item()-.3) < 1e-6
    data.body_pos_w[0, 0, 2] = .45
    data.body_pos_w[0, 1:, 2] = .205
    forces[0, 1:, 2] = 20.
    # At tread height but behind its crossing plane is NOT success.
    for _ in range(12):
        term._update_command()
    assert term.phase.item() == 2
    assert term.success.item() == 0
    data.root_pos_w[0, 0] = .35
    data.body_pos_w[0, 1:, 0] = .35
    for _ in range(10):
        term._update_command()
    assert term.phase.item() == 0
    assert term.success.item() == 1
    assert abs(term.height_command.item()-.35) < 1e-6
    assert term.completed_steps.item() == 1
    assert term.course_success.item() == 0
    # Retreating to the same tread must not start a new paid attempt.
    data.root_pos_w[0, 0] = 0.
    for _ in range(30):
        term._update_command()
    assert term.phase.item() == 0
    assert term.success.item() == 0
    # Approach and complete the next distinct tread.
    data.root_pos_w[0, 0] = .8
    term._env.scene['terrain_scan'].data.ray_hits_w[:] = torch.tensor([[[1.1, 0., .35]]])
    term.reference_ground_height = lambda: torch.tensor([.15])
    term._update_command()
    assert term.phase.item() == 1
    data.root_pos_w[0, 0] = 1.2
    data.body_pos_w[0, 1:, 0] = 1.2
    data.body_pos_w[0, 1:, 2] = .405
    data.body_pos_w[0, 0, 2] = .65
    for _ in range(10):
        term._update_command()
    assert term.completed_steps.item() == 2
    assert term.course_success.item() == 1
    assert scope['ascent_completed'](term._env).item()
    assert scope['ascent_course_success'](term._env).item() * term._env.step_dt == 1.
    term._update_command()
    assert term.course_success.item() == 0
    # Repeated consumers in one physics step must not clear an event or add time.
    elapsed = term.elapsed.clone()
    term.check_state()
    term.check_state()
    assert torch.equal(term.elapsed, elapsed)
    # Episode reset clears rewards/history but retains curriculum streaks.
    term.success_streak[:] = 2
    logs = term.reset(torch.tensor([0]))
    assert logs['two_step_pass_rate'] == 1.
    assert term.completed_steps.item() == 0
    assert torch.isneginf(term.highest_completed_top).all()
    assert term.success_streak.item() == 2


def test_lift_rewards_do_not_repeat_on_oscillation(ascent_fixture):
    term, data, _ = ascent_fixture
    term._update_command()
    data.body_pos_w[0, 1:, 0] = .1
    data.body_pos_w[0, 1:, 2] = .105
    data.body_pos_w[0, 0, 2] = .40
    term._update_command()
    assert term.wheel_lift.item() > 0
    assert term.body_lift.item() > 0
    for _ in range(3):
        data.body_pos_w[0, 1:, 2] = .055
        data.body_pos_w[0, 0, 2] = .35
        term._update_command()
        assert term.wheel_lift.item() == term.body_lift.item() == 0
        data.body_pos_w[0, 1:, 2] = .105
        data.body_pos_w[0, 0, 2] = .40
        term._update_command()
        assert term.wheel_lift.item() == term.body_lift.item() == 0


def test_approach_gets_more_than_four_seconds_but_stall_ends(ascent_fixture):
    term, data, _ = ascent_fixture
    term._env.scene['terrain_scan'].data.ray_hits_w[0, 0, 0] = .6
    for _ in range(220):
        term._update_command()
    assert term.phase.item() == 1
    assert term.failed.item() == 0
    assert term.near_time.item() == 0
    data.body_pos_w[0, 1:, 0] = .5
    for _ in range(160):
        term._update_command()
        if term.failed.item():
            break
    assert term.failed.item() == 1
    assert term.phase.item() == 0


def test_climbing_pitch_debounce_and_severe_tip(ascent_fixture):
    term, data, _ = ascent_fixture
    term._update_command()
    term._env.command_manager.get_term = lambda name: term
    def tilt(pitch):
        data.projected_gravity_b[:] = torch.tensor([[torch.sin(torch.tensor(pitch)), 0.,
                                                    -torch.cos(torch.tensor(pitch))]])
    tilt(.85)  # 49 degrees is permitted for an active attempt.
    assert not scope['ascent_bad_roll_pitch'](term._env, .7, NS(name='robot')).item()
    tilt(1.1)
    term._env.common_step_counter += 1
    assert not scope['ascent_bad_roll_pitch'](term._env, .7, NS(name='robot')).item()
    for _ in range(9):
        term._env.common_step_counter += 1
        result = scope['ascent_bad_roll_pitch'](term._env, .7, NS(name='robot'))
    assert result.item()
    term.tilt_time.zero_()
    tilt(1.4)
    assert scope['ascent_bad_roll_pitch'](term._env, .7, NS(name='robot')).item()


def test_curriculum_needs_full_course_success():
    term = NS(completed_steps=torch.tensor([2, 1]), success_streak=torch.zeros(2),
              failure_streak=torch.zeros(2))
    terrain = NS(terrain_types=torch.tensor([3, 3]), terrain_levels=torch.tensor([2, 2]),
                 cfg=NS(terrain_generator=NS(num_rows=10, num_cols=20,
                    sub_terrains={'pyramid_stairs': NS(proportion=.15),
                                  'pyramid_stairs_inv': NS(proportion=.85)})),
                 terrain_origins=torch.zeros(10, 20, 3), env_origins=torch.zeros(2, 3))
    env = NS(num_envs=2, device='cpu', scene=NS(terrain=terrain),
             command_manager=NS(get_term=lambda name: term),
             termination_manager=NS(terminated=torch.zeros(2, dtype=torch.bool)))
    for _ in range(3):
        scope['update_ascent_curriculum'](env, torch.arange(2))
    assert terrain.terrain_levels.tolist() == [3, 1]
