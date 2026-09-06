"""Compatibility with the legacy Isaac Sim MJCF equality importer."""

from dataclasses import fields
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MjcfConverter
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


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
        stage.GetRootLayer().Save()


def spawn_closed_loop_mjcf(prim_path, cfg, translation=None, orientation=None):
    converter = ClosedLoopMjcfConverter(cfg)
    properties = {
        field.name: getattr(cfg, field.name)
        for field in fields(sim_utils.UsdFileCfg)
        if field.name not in {"func", "usd_path"} and hasattr(cfg, field.name)
    }
    usd_cfg = sim_utils.UsdFileCfg(usd_path=converter.usd_path, **properties)
    return sim_utils.spawn_from_usd(prim_path, usd_cfg, translation, orientation)
