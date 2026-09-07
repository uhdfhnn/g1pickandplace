from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"


def _load_minimal_scene_helper():
    """Load the configuration helper with simulator-free stand-ins."""

    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    constant_names = {
        "MINIMAL_DEMO_PUBLIC_OBJECT_SIZE_M",
        "MINIMAL_DEMO_OBJECT_SIZE_M",
        "MINIMAL_DEMO_PUBLIC_OBJECT_CENTER_Z_M",
        "MINIMAL_DEMO_DESK_TOP_Z_M",
        "MINIMAL_DEMO_OBJECT_CENTER_Z_M",
        "MINIMAL_DEMO_GOAL_SUPPORT_SIZE_M",
        "MINIMAL_DEMO_ROBOT_JOINT_SEED_RAD",
        "MINIMAL_DEMO_TABLE_VISUAL_COLOR_RGB",
        "MINIMAL_DEMO_BACKDROP_POSITION_WORLD_M",
        "MINIMAL_DEMO_BACKDROP_SIZE_M",
        "MINIMAL_DEMO_BACKDROP_COLOR_RGB",
        "MINIMAL_DEMO_BACKDROP_COLLISION_ENABLED",
        "MINIMAL_DEMO_DOME_LIGHT_COLOR_RGB",
        "MINIMAL_DEMO_DOME_LIGHT_INTENSITY",
    }
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in constant_names
            for target in node.targets
        )
    ]
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_configure_minimal_demo_scene"
    )
    safe_reset_helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_safe_reset_posture_config"
    )
    namespace = {"Any": object, "Mapping": Mapping}
    exec(
        compile(
            ast.Module(body=assignments + [safe_reset_helper, helper], type_ignores=[]),
            str(SCRIPT),
            "exec",
        ),
        namespace,
    )
    return namespace


class _FakeConfig:
    """Small config stand-in that keeps constructor fields inspectable."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeCuboid(_FakeConfig):
    def __init__(self, size, **kwargs):
        super().__init__(size=size, **kwargs)


class _FakeAssetBase(_FakeConfig):
    class InitialStateCfg(_FakeConfig):
        pass


class _FakePreviewSurface(_FakeConfig):
    pass


class _FakeCollisionProperties(_FakeConfig):
    pass


class _FakeDomeLight(_FakeConfig):
    pass


def _fake_configuration():
    namespace = _load_minimal_scene_helper()
    namespace["AssetBaseCfg"] = _FakeAssetBase
    namespace["sim_utils"] = SimpleNamespace(
        CuboidCfg=_FakeCuboid,
        CollisionPropertiesCfg=_FakeCollisionProperties,
        PreviewSurfaceCfg=_FakePreviewSurface,
        DomeLightCfg=_FakeDomeLight,
    )
    namespace["args"] = SimpleNamespace(target_position=(-4.15, -4.03, 0.824))
    table_spawn = _FakeConfig(
        usd_path="/public/assets/objects/table_with_yellowbox.usd",
        visual_material="authored-material",
    )
    object_cfg = SimpleNamespace(
        spawn=_FakeCuboid((0.06, 0.06, 0.06)),
        init_state=SimpleNamespace(pos=[-4.25, -4.03, 0.84], rot=[1.0, 0.0, 0.0, 0.0]),
    )
    robot_joint_pos = {
        "left_shoulder_pitch_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "left_elbow_joint": 0.0,
        "right_elbow_joint": 0.0,
        "waist_yaw_joint": 0.0,
    }
    robot_cfg = SimpleNamespace(
        init_state=SimpleNamespace(joint_pos=robot_joint_pos),
    )
    scene = SimpleNamespace(
        room_walls=object(),
        packing_table=_FakeAssetBase(
            prim_path="/World/envs/env_.*/PackingTable",
            init_state=_FakeAssetBase.InitialStateCfg(
                pos=[-4.3, -4.2, -0.2],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=table_spawn,
        ),
        light=None,
        object=object_cfg,
        robot=robot_cfg,
    )
    env_cfg = SimpleNamespace(scene=scene)
    return namespace, env_cfg, robot_joint_pos


def test_minimal_scene_constants_and_reset_configuration_are_exact():
    namespace, env_cfg, robot_joint_pos = _fake_configuration()
    configure = namespace["_configure_minimal_demo_scene"]

    payload = configure(env_cfg)

    assert namespace["MINIMAL_DEMO_OBJECT_SIZE_M"] == (0.04, 0.04, 0.04)
    assert namespace["MINIMAL_DEMO_GOAL_SUPPORT_SIZE_M"] == (0.08, 0.06, 0.01)
    assert env_cfg.scene.room_walls is None
    assert env_cfg.scene.packing_table.spawn.usd_path == "/public/assets/objects/table_with_yellowbox.usd"
    assert env_cfg.scene.packing_table.init_state.pos == [-4.3, -4.2, -0.2]
    assert env_cfg.scene.packing_table.init_state.rot == [1.0, 0.0, 0.0, 0.0]
    assert isinstance(env_cfg.scene.packing_table.spawn.visual_material, _FakePreviewSurface)
    assert (
        env_cfg.scene.packing_table.spawn.visual_material.diffuse_color
        == namespace["MINIMAL_DEMO_TABLE_VISUAL_COLOR_RGB"]
    )
    assert env_cfg.scene.object.spawn.size == (0.04, 0.04, 0.04)
    assert namespace["MINIMAL_DEMO_DESK_TOP_Z_M"] == 0.794
    assert env_cfg.scene.object.init_state.pos == (-4.25, -4.03, 0.814)
    assert env_cfg.scene.object.init_state.rot == [1.0, 0.0, 0.0, 0.0]
    assert payload["goal_support_center_world_m"] == pytest.approx([-4.15, -4.03, 0.799])
    assert payload["desk_top_world_z_m"] == 0.794
    assert payload["robot_initial_joint_seed_rad"] == {
        "left_shoulder_pitch_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.15,
        "right_shoulder_roll_joint": -0.15,
        "left_elbow_joint": 0.10,
        "right_elbow_joint": 0.10,
    }
    assert robot_joint_pos["waist_yaw_joint"] == 0.0

    backdrop = env_cfg.scene.minimal_demo_backdrop
    assert backdrop.prim_path == "{ENV_REGEX_NS}/MinimalDemoBackdrop"
    assert backdrop.init_state.pos == namespace["MINIMAL_DEMO_BACKDROP_POSITION_WORLD_M"]
    assert backdrop.spawn.size == namespace["MINIMAL_DEMO_BACKDROP_SIZE_M"]
    assert backdrop.spawn.visual_material.diffuse_color == namespace["MINIMAL_DEMO_BACKDROP_COLOR_RGB"]
    assert backdrop.spawn.collision_props.collision_enabled is False
    # The real CuboidCfg inherits ``rigid_props=None`` from its public
    # RigidObjectSpawnerCfg base; the simulator-free stand-in only exposes
    # fields passed by the helper, so use the same effective default here.
    assert getattr(backdrop.spawn, "rigid_props", None) is None

    dome_light = env_cfg.scene.minimal_demo_dome_light
    assert dome_light.prim_path == "{ENV_REGEX_NS}/MinimalDemoDomeLight"
    assert dome_light.spawn.color == namespace["MINIMAL_DEMO_DOME_LIGHT_COLOR_RGB"]
    assert dome_light.spawn.intensity == namespace["MINIMAL_DEMO_DOME_LIGHT_INTENSITY"]
    assert payload["visual_assets"] == {
        "table_scene_name": "packing_table",
        "backdrop_scene_name": "minimal_demo_backdrop",
        "dome_light_scene_name": "minimal_demo_dome_light",
    }


def test_minimal_scene_preserves_an_existing_public_light():
    namespace, env_cfg, _ = _fake_configuration()
    existing_light = _FakeAssetBase(
        prim_path="/World/light",
        spawn=_FakeDomeLight(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    env_cfg.scene.light = existing_light

    payload = namespace["_configure_minimal_demo_scene"](env_cfg)

    assert env_cfg.scene.light is existing_light
    assert not hasattr(env_cfg.scene, "minimal_demo_dome_light")
    assert payload["visual_assets"]["dome_light_scene_name"] == "light"


@pytest.mark.parametrize(
    "bad_object",
    [
        SimpleNamespace(spawn=object(), init_state=SimpleNamespace(pos=[0.0, 0.0, 0.84], rot=[])),
        SimpleNamespace(
            spawn=_FakeCuboid((0.06, 0.06, 0.06)),
            init_state=SimpleNamespace(pos=None, rot=[]),
        ),
    ],
)
def test_minimal_scene_rejects_unknown_object_configuration(bad_object):
    namespace, env_cfg, _ = _fake_configuration()
    env_cfg.scene.object = bad_object
    with pytest.raises(RuntimeError, match="minimal-demo-scene"):
        namespace["_configure_minimal_demo_scene"](env_cfg)


def test_minimal_scene_flag_is_explicit_store_true_and_defaults_false():
    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    matching_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and node.args[0].value == "--minimal-demo-scene":
            matching_calls.append(node)

    assert len(matching_calls) == 1
    call = matching_calls[0]
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords["action"], ast.Constant)
    assert keywords["action"].value == "store_true"
    assert "default" not in keywords


def test_runtime_diagnostics_expose_minimal_scene_configuration():
    source = SCRIPT.read_text()
    assert '"minimal_demo_scene"' in source
    assert '"minimal_demo_configuration"' in source
    assert 'collision_enabled=True' in source


def _load_reset_posture_validator():
    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id
                in {
                    "MINIMAL_DEMO_ROBOT_JOINT_SEED_RAD",
                    "MINIMAL_DEMO_RESET_POSTURE_MAX_ERROR_RAD",
                }
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "_validate_minimal_demo_reset_posture"
        )
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


def test_live_reset_posture_gate_accepts_small_error_and_rejects_old_pose():
    namespace = _load_reset_posture_validator()
    validate = namespace["_validate_minimal_demo_reset_posture"]
    seed = namespace["MINIMAL_DEMO_ROBOT_JOINT_SEED_RAD"]
    names = tuple(seed)
    defaults = np.asarray([seed[name] for name in names])

    errors = validate(defaults + 0.01, defaults, names)
    assert max(errors.values()) == pytest.approx(0.01)
    with pytest.raises(RuntimeError, match="did not reach configured defaults"):
        validate(np.zeros_like(defaults), defaults, names)


def test_reset_state_write_occurs_before_ik_construction():
    source = SCRIPT.read_text()
    assert source.index("_write_minimal_demo_reset_posture(env)") < source.index(
        'print("[plan] constructing the public Pinocchio URDF model"'
    )
