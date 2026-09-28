"""Strict 49D flat -> appended terrain-input transfer for installed RSL-RL MLPs."""
import torch


def expand_state(source, target, old_dim=49):
    if source.keys() != target.keys():
        raise ValueError("Checkpoint model keys differ; transfer requires matching MLP architecture")
    result = {}
    for name, value in source.items():
        expected = target[name]
        if name == "mlp.0.weight":
            if value.shape[1] != old_dim or expected.shape[1] <= old_dim or value.shape[0] != expected.shape[0]:
                raise ValueError(f"Invalid input expansion: {value.shape} -> {expected.shape}")
            result[name] = torch.zeros_like(expected)
            result[name][:, :old_dim] = value.to(expected)
        elif value.shape != expected.shape:
            raise ValueError(f"Unexpected parameter mismatch for {name}")
        else:
            result[name] = value.to(expected)
    return result


def transfer_flat(runner, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # Normalize input order is part of the contract, not inferred from tensor width alone.
    expected_terms = ["base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
                      "height_command", "base_height", "base_height_error", "joint_pos", "joint_vel", "actions",
                      "terrain_scan"]
    actual = runner.env.unwrapped.observation_manager.active_terms["policy"]
    if list(actual) != expected_terms:
        raise ValueError(f"Unexpected observation order: {actual}")
    for label in ("actor", "critic"):
        model = getattr(runner.alg, label)
        if model.obs_normalization:
            raise ValueError("This transfer expects unnormalized observations")
        model.load_state_dict(expand_state(checkpoint[f"{label}_state_dict"], model.state_dict()), strict=True)
    runner.alg.optimizer.state.clear()
    runner.current_learning_iteration = 0
    print("[INFO] Transferred actor/critic: old 49 input columns copied, added columns zero; fresh optimizer/iteration.")
