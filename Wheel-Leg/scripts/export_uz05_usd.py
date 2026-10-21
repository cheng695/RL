"""Export the UZ-05 MJCF as a reusable USD asset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_SOURCE = PROJECT_ROOT / "source" / "my_robot_lab"
if str(EXTENSION_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXTENSION_SOURCE))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export the UZ-05 robot to USD.")
parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_ROOT / "assets" / "uz05" / "uz05.usd",
    help="Output USD path (default: Wheel-Leg/assets/uz05/uz05.usd).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.utils.extensions import enable_extension
import isaaclab.sim as sim_utils

from my_robot_lab.assets.robots.mjcf import ClosedLoopMjcfConverter


def main() -> None:
    enable_extension("isaacsim.asset.importer.mjcf")
    output_path = args_cli.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mjcf_path = (
        EXTENSION_SOURCE
        / "my_robot_lab"
        / "assets"
        / "robots"
        / "UZ05_MJCF_real_params"
        / "xmls"
        / "uz05_isaac.xml"
    )
    cfg = sim_utils.MjcfFileCfg(
        asset_path=str(mjcf_path),
        usd_dir=str(output_path.parent),
        usd_file_name=output_path.name,
        force_usd_conversion=True,
        make_instanceable=True,
        import_sites=True,
        fix_base=False,
        self_collision=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    )
    converter = ClosedLoopMjcfConverter(cfg)
    generated_path = Path(converter.usd_path).resolve()
    if generated_path != output_path:
        raise RuntimeError(
            f"Converter wrote {generated_path}, expected requested output {output_path}. "
            "Check the Isaac Lab version's MJCF converter output settings."
        )
    print(f"[INFO] Exported UZ-05 USD: {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
