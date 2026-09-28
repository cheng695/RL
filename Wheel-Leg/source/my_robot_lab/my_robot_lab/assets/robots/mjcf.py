"""Compatibility with the legacy Isaac Sim MJCF equality importer."""

from dataclasses import fields
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MjcfConverter
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


class ClosedLoopMjcfConverter(MjcfConverter):
    """Import the tree officially and author connect constraints as USD joints."""

    def _convert_asset(self, cfg):
        source = Path(cfg.asset_path).absolute()
        tree = ET.parse(source)
        root = tree.getroot()
        equality = root.find("equality")
        if equality is None:
            return super()._convert_asset(cfg)
        if any(term.tag != "connect" for term in equality):
            raise ValueError("Only connect equality constraints are supported by this adapter.")
        root.remove(equality)

        with tempfile.TemporaryDirectory(prefix="uz05_mjcf_") as directory:
            compiler = root.find("compiler")
            for attribute in ("meshdir", "texturedir"):
                asset_dir = source.parent / compiler.get(attribute, ".")
                compiler.set(attribute, os.path.relpath(asset_dir, directory))
            temporary_source = Path(directory) / source.name
            tree.write(temporary_source, encoding="utf-8", xml_declaration=True)
            super()._convert_asset(cfg.replace(asset_path=str(temporary_source)))

        stage = Usd.Stage.Open(self.usd_path)
        if not stage or not stage.GetDefaultPrim():
            raise RuntimeError(f"MJCF conversion did not create a valid robot: {source}")
        world_body = stage.GetPrimAtPath(stage.GetDefaultPrim().GetPath().AppendChild("worldBody"))
        if world_body and not world_body.GetChildren():
            world_body.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            world_body.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        bodies = {prim.GetName(): prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
        transforms = UsdGeom.XformCache()
        for term in equality:
            body0 = bodies[term.attrib["body1"]]
            body1 = bodies[term.attrib["body2"]]
            anchor0 = Gf.Vec3d(*map(float, term.attrib["anchor"].split()))
            world_anchor = transforms.GetLocalToWorldTransform(body0).Transform(anchor0)
            anchor1 = transforms.GetLocalToWorldTransform(body1).GetInverse().Transform(world_anchor)
            path = stage.GetDefaultPrim().GetPath().AppendPath("loop_joints/" + term.attrib["name"])
            joint = UsdPhysics.SphericalJoint.Define(stage, path)
            joint.CreateBody0Rel().SetTargets([body0.GetPath()])
            joint.CreateBody1Rel().SetTargets([body1.GetPath()])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(anchor0))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(anchor1))
            joint.CreateExcludeFromArticulationAttr().Set(True)
        self._restore_reference_inertials(stage, source, bodies)
        stage.GetRootLayer().Save()

    @staticmethod
    def _restore_reference_inertials(stage, source, bodies):
        """Use compiled MuJoCo inertials; fuse massless fixed frames into chassis.

        A USD mass of zero requests automatic mass calculation, not a massless
        body. Remove the rigid-body schemas instead, and rebase attached joints.
        Geometry and named Xforms remain in place under the chassis rigid body.
        """
        reference = mujoco.MjModel.from_xml_path(str(source.with_name("uz05.xml")))
        chassis = bodies["chassis"]
        frames = {name: bodies[name] for name in (
            "legacy_y_forward_model_frame", "left_leg_mount", "right_leg_mount"
        )}
        transforms = UsdGeom.XformCache()
        chassis_world = transforms.GetLocalToWorldTransform(chassis)
        redirects = {str(prim.GetPath()): prim for prim in frames.values()}
        remove_joints = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            for side in (0, 1):
                relation = getattr(joint, f"GetBody{side}Rel")()
                targets = relation.GetTargets()
                if len(targets) != 1 or str(targets[0]) not in redirects:
                    continue
                old = redirects[str(targets[0])]
                relative = transforms.GetLocalToWorldTransform(old) * chassis_world.GetInverse()
                pos_attr = getattr(joint, f"GetLocalPos{side}Attr")()
                rot_attr = getattr(joint, f"GetLocalRot{side}Attr")()
                local = Gf.Matrix4d(1.0)
                local.SetRotate(Gf.Quatd(rot_attr.Get()))
                local.SetTranslateOnly(Gf.Vec3d(pos_attr.Get()))
                rebased = local * relative
                pos_attr.Set(Gf.Vec3f(rebased.ExtractTranslation()))
                rot_attr.Set(Gf.Quatf(rebased.ExtractRotationQuat()))
                relation.SetTargets([chassis.GetPath()])
            if joint.GetBody0Rel().GetTargets() == joint.GetBody1Rel().GetTargets():
                if not prim.IsA(UsdPhysics.FixedJoint):
                    raise RuntimeError(f"Cannot fuse non-fixed joint: {prim.GetPath()}")
                remove_joints.append(prim.GetPath())
        for path in remove_joints:
            # The importer may author joints through a referenced USD layer;
            # deleting a local spec alone can expose the original joint again.
            stage.GetPrimAtPath(path).SetActive(False)
        for prim in frames.values():
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            prim.RemoveAPI(UsdPhysics.MassAPI)
            relative = transforms.GetLocalToWorldTransform(prim) * chassis_world.GetInverse()
            # Spatial tendon roots must live on the surviving rigid body.
            schemas = list(prim.GetAppliedSchemas())
            for schema in schemas:
                if not schema.startswith("PhysxTendonAttachment"):
                    continue
                schema_type, instance = schema.split(":", 1)
                # GetAppliedSchemas includes inherited base APIs. Applying both
                # base and Root/Leaf creates duplicate attachment instances.
                if schema_type == "PhysxTendonAttachmentAPI" and any(
                    s in schemas for s in (
                        f"PhysxTendonAttachmentRootAPI:{instance}",
                        f"PhysxTendonAttachmentLeafAPI:{instance}",
                    )
                ):
                    continue
                getattr(PhysxSchema, schema_type).Apply(chassis, instance)
                for attr in prim.GetAttributes():
                    if instance not in attr.GetName().split(":") or not attr.HasAuthoredValueOpinion():
                        continue
                    value = attr.Get()
                    if attr.GetName().endswith(":localPos"):
                        value = Gf.Vec3f(relative.Transform(Gf.Vec3d(value)))
                    chassis.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(value)
                prim.RemoveAPI(getattr(PhysxSchema, schema_type), instance)
        for prim in stage.Traverse():
            for relation in prim.GetRelationships():
                if "tendon" not in relation.GetName().lower():
                    continue
                targets = relation.GetTargets()
                if any(str(path) in redirects for path in targets):
                    relation.SetTargets([chassis.GetPath() if str(path) in redirects else path for path in targets])
        # Imported bodies are siblings, not children of the chassis rigid body.
        # Move their geometry below chassis; just removing RigidBodyAPI would
        # leave these colliders attached to the world as static obstacles.
        # Deinstance these small subtrees before copying: flattened instances
        # otherwise reference anonymous prototype roots outside the subtree.
        for prim in frames.values():
            while True:
                instances = [p for p in Usd.PrimRange(prim) if p.IsInstance()]
                if not instances:
                    break
                for instance in instances:
                    instance.SetInstanceable(False)
        flattened = stage.Flatten()
        for name, prim in frames.items():
            for descendant in Usd.PrimRange(prim):
                if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise RuntimeError(f"Unexpected nested body in frame {name}")
            destination = chassis.GetPath().AppendChild(name)
            Sdf.CopySpec(flattened, prim.GetPath(), stage.GetRootLayer(), destination)
            moved = stage.GetPrimAtPath(destination)
            relative = transforms.GetLocalToWorldTransform(prim) * chassis_world.GetInverse()
            UsdGeom.Xformable(moved).MakeMatrixXform().Set(relative)
            UsdGeom.Xformable(moved).SetResetXformStack(False)
            prim.SetActive(False)
        total = 0.0
        for name, prim in bodies.items():
            if name in frames:
                continue
            body = reference.body(name)
            index = body.id
            mass = float(reference.body_mass[index])
            if mass <= 0:
                raise RuntimeError(f"Unexpected massless rigid body: {name}")
            api = UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr().Set(mass)
            api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*map(float, reference.body_ipos[index])))
            api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, reference.body_inertia[index])))
            w, x, y, z = map(float, reference.body_iquat[index])
            api.CreatePrincipalAxesAttr().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
            total += mass
        expected = float(reference.body_mass.sum())
        if abs(total - expected) > 1e-6:
            raise RuntimeError(f"Incomplete inertial mapping: {total} != {expected}")
        print(f"[INFO] Restored MuJoCo inertials: {total:.6f} kg; fused three massless frames.")


def spawn_closed_loop_mjcf(prim_path, cfg, translation=None, orientation=None):
    converter = ClosedLoopMjcfConverter(cfg)
    properties = {
        field.name: getattr(cfg, field.name)
        for field in fields(sim_utils.UsdFileCfg)
        if field.name not in {"func", "usd_path"} and hasattr(cfg, field.name)
    }
    usd_cfg = sim_utils.UsdFileCfg(usd_path=converter.usd_path, **properties)
    return sim_utils.spawn_from_usd(prim_path, usd_cfg, translation, orientation)
