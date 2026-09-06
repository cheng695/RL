from pathlib import Path

import mujoco
import mujoco.viewer


XML_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "my_robot_lab"
    / "my_robot_lab"
    / "assets"
    / "robots"
    / "UZ05_MJCF_real_params"
    / "xmls"
    / "uz05.xml"
)


def build_view_xml() -> Path:
    xml = XML_PATH.read_text()

    scene = """
    <light name="key_light" pos="0 -3 3" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <light name="fill_light" pos="3 3 2" dir="-1 -1 -0.5" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="5 5 0.05" pos="0 0 0"
          rgba="0.25 0.25 0.25 1" contype="1" conaffinity="1"/>
"""

    xml = xml.replace("<worldbody>", "<worldbody>" + scene, 1)

    # Keep the generated XML next to the source XML so meshdir="../meshes/stl" still resolves.
    view_xml_path = XML_PATH.with_name("uz05_view.xml")
    view_xml_path.write_text(xml)
    return view_xml_path


def print_model_info(model: mujoco.MjModel) -> None:
    print("MuJoCo:", mujoco.__version__)
    print("xml:", XML_PATH)
    print("nq:", model.nq, "nv:", model.nv, "nu:", model.nu)

    print("\nJoints:")
    for joint_id in range(model.njnt):
        joint = model.joint(joint_id)
        limited = bool(model.jnt_limited[joint_id])
        joint_range = model.jnt_range[joint_id]
        print(f"  {joint.name}: limited={limited}, range=({joint_range[0]:.4f}, {joint_range[1]:.4f})")

    print("\nActuators:")
    for actuator_id in range(model.nu):
        actuator = model.actuator(actuator_id)
        ctrl_range = model.actuator_ctrlrange[actuator_id]
        force_range = model.actuator_forcerange[actuator_id]
        print(
            f"  {actuator.name}: ctrl=({ctrl_range[0]:.4f}, {ctrl_range[1]:.4f}), "
            f"force=({force_range[0]:.4f}, {force_range[1]:.4f})"
        )

    print("\nBodies:")
    for body_id in range(model.nbody):
        print(f"  {body_id}: {model.body(body_id).name}")

    print("\nTendons:")
    for tendon_id in range(model.ntendon):
        tendon = model.tendon(tendon_id)
        limited = bool(model.tendon_limited[tendon_id])
        tendon_range = model.tendon_range[tendon_id]
        print(f"  {tendon.name}: limited={limited}, range=({tendon_range[0]:.4f}, {tendon_range[1]:.4f})")


def main() -> None:
    view_xml_path = build_view_xml()
    model = mujoco.MjModel.from_xml_path(str(view_xml_path))
    data = mujoco.MjData(model)

    print_model_info(model)
    print("\nOpening viewer. Press ESC to close.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
