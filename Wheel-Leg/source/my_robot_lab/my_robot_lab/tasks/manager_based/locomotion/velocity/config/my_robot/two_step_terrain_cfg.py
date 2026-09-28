"""Fixed 150 mm then 200 mm risers, measured outward from the spawn platform."""
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass
from ...mdp.two_step_terrain import two_step_up


@configclass
class TwoStepTerrainCfg(SubTerrainBaseCfg):
    function = two_step_up
    first_step_height: float = .15
    second_step_height: float = .20
    platform_width: float = 3.
    step_width: float = .8
