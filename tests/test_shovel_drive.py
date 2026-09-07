from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from pathlib import Path
import re
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"


def _load_actuator_helpers() -> dict[str, object]:
    """Extract the actuator helpers without importing Isaac Sim or Kit."""

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    assignment_names = {
        "DEX1_LEFT_GRIPPER_JOINT_NAMES",
        "DEX1_RIGHT_GRIPPER_JOINT_NAMES",
        "SHOVEL_HAND_ACTUATOR_GROUP_NAME",
        "SHOVEL_LEFT_HAND_STIFFNESS_NM_PER_RAD",
        "SHOVEL_LEFT_HAND_DAMPING_NM_S_PER_RAD",
        "SHOVEL_RIGHT_HAND_STIFFNESS_NM_PER_RAD",
        "SHOVEL_RIGHT_HAND_DAMPING_NM_S_PER_RAD",
        "SHOVEL_HAND_ACTUATOR_RUNTIME_ATOL",
    }
    function_names = {
        "_shovel_hand_actuator_targets",
        "_actuator_parameter_for_joint",
        "_configure_shovel_hand_actuator",
        "_verify_shovel_hand_actuator_runtime",
    }
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in assignment_names
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name in function_names)
    ]
    namespace: dict[str, object] = {
        "Any": object,
        "Mapping": Mapping,
        "math": math,
        "re": re,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


def _public_joint_names(namespace: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        namespace["DEX1_LEFT_GRIPPER_JOINT_NAMES"]  # type: ignore[arg-type]
    ) + tuple(
        namespace["DEX1_RIGHT_GRIPPER_JOINT_NAMES"]  # type: ignore[arg-type]
    )


class _FakeHands:
    def __init__(self, *, joint_names_expr, stiffness=800.0, damping=3.0) -> None:
        self.joint_names_expr = joint_names_expr
        self.stiffness = stiffness
        self.damping = damping
        # These fields are deliberately unrelated to the gain override.  The
        # public Isaac Lab actuator contract owns their upstream values, and a
        # shovel calibration must leave effort/velocity/friction unchanged.
        self.effort_limit = "upstream-effort"
        self.velocity_limit = "upstream-velocity"
        self.friction = "upstream-friction"


def _fake_scene(namespace: Mapping[str, object], hands: _FakeHands):
    group_name = namespace["SHOVEL_HAND_ACTUATOR_GROUP_NAME"]
    return SimpleNamespace(
        robot=SimpleNamespace(actuators={group_name: hands}),
    )


def test_config_override_changes_left_hand_only_and_returns_payload() -> None:
    namespace = _load_actuator_helpers()
    joints = _public_joint_names(namespace)
    hands = _FakeHands(joint_names_expr=joints)
    scene = _fake_scene(namespace, hands)

    payload = namespace["_configure_shovel_hand_actuator"](scene)  # type: ignore[operator]

    assert hands.stiffness == {
        joints[0]: 1600.0,
        joints[1]: 1600.0,
        joints[2]: 800.0,
        joints[3]: 800.0,
    }
    assert hands.damping == {
        joints[0]: 6.0,
        joints[1]: 6.0,
        joints[2]: 3.0,
        joints[3]: 3.0,
    }
    assert hands.effort_limit == "upstream-effort"
    assert hands.velocity_limit == "upstream-velocity"
    assert hands.friction == "upstream-friction"
    assert payload["status"] == "PASS"
    assert payload["group"] == "hands"
    assert payload["joint_names"] == list(joints)
    assert payload["stiffness_Nm_per_rad"] == hands.stiffness
    assert payload["damping_Nm_s_per_rad"] == hands.damping
    assert payload["upstream_stiffness_Nm_per_rad"] == dict.fromkeys(joints, 800.0)
    assert payload["upstream_damping_Nm_s_per_rad"] == dict.fromkeys(joints, 3.0)


@pytest.mark.parametrize(
    "scene_factory",
    [
        lambda namespace: SimpleNamespace(),
        lambda namespace: SimpleNamespace(robot=SimpleNamespace()),
        lambda namespace: SimpleNamespace(robot=SimpleNamespace(actuators={})),
        lambda namespace: _fake_scene(
            namespace,
            _FakeHands(
                joint_names_expr=tuple(
                    _public_joint_names(namespace) + ("unexpected_joint",)
                )
            ),
        ),
        lambda namespace: _fake_scene(
            namespace,
            _FakeHands(
                joint_names_expr=_public_joint_names(namespace),
                stiffness=801.0,
            ),
        ),
        lambda namespace: _fake_scene(
            namespace,
            _FakeHands(
                joint_names_expr=_public_joint_names(namespace),
                damping=3.1,
            ),
        ),
    ],
)
def test_config_override_fails_closed_for_missing_or_mismatched_contract(scene_factory) -> None:
    namespace = _load_actuator_helpers()
    scene = scene_factory(namespace)

    with pytest.raises(RuntimeError, match="Task-4 shovel"):
        namespace["_configure_shovel_hand_actuator"](scene)  # type: ignore[operator]


def _runtime_env(namespace: Mapping[str, object], *, mismatch: bool = False):
    joints = _public_joint_names(namespace)
    stiffness = [0.0] * len(joints)
    damping = [0.0] * len(joints)
    targets = namespace["_shovel_hand_actuator_targets"]()  # type: ignore[operator]
    expected_stiffness, expected_damping = targets
    for index, joint in enumerate(joints):
        stiffness[index] = expected_stiffness[joint]
        damping[index] = expected_damping[joint]
    if mismatch:
        # Deliberately exceed the 1e-6 absolute runtime gate by one order of
        # magnitude; this models a real PhysX/config conversion discrepancy.
        stiffness[0] += 1.0e-5
    data = SimpleNamespace(
        joint_names=joints,
        # Isaac Lab uses [environment, joint] tensors; nested Python lists are
        # the dependency-light equivalent consumed by the runtime helper.
        joint_stiffness=[stiffness],
        joint_damping=[damping],
    )
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})


def test_runtime_verification_reads_live_values_and_passes() -> None:
    namespace = _load_actuator_helpers()
    report = namespace["_verify_shovel_hand_actuator_runtime"](  # type: ignore[operator]
        _runtime_env(namespace)
    )

    assert report["status"] == "PASS"
    assert report["source"] == "env.scene['robot'].data"
    assert report["checked_joint_names"] == list(_public_joint_names(namespace))
    assert report["stiffness_Nm_per_rad"] == {
        _public_joint_names(namespace)[0]: 1600.0,
        _public_joint_names(namespace)[1]: 1600.0,
        _public_joint_names(namespace)[2]: 800.0,
        _public_joint_names(namespace)[3]: 800.0,
    }
    assert report["damping_Nm_s_per_rad"] == {
        _public_joint_names(namespace)[0]: 6.0,
        _public_joint_names(namespace)[1]: 6.0,
        _public_joint_names(namespace)[2]: 3.0,
        _public_joint_names(namespace)[3]: 3.0,
    }


def test_runtime_verification_fails_closed_on_live_gain_mismatch() -> None:
    namespace = _load_actuator_helpers()

    with pytest.raises(RuntimeError, match="runtime actuator gains"):
        namespace["_verify_shovel_hand_actuator_runtime"](  # type: ignore[operator]
            _runtime_env(namespace, mismatch=True)
        )


def test_runtime_verification_fails_closed_when_expected_joint_is_absent() -> None:
    namespace = _load_actuator_helpers()
    env = _runtime_env(namespace)
    env.scene["robot"].data.joint_names = _public_joint_names(namespace)[:-1]

    with pytest.raises(RuntimeError, match="missing expected joints"):
        namespace["_verify_shovel_hand_actuator_runtime"](env)  # type: ignore[operator]


def test_runtime_verifier_is_shovel_only_diagnostic_path() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    diagnostics = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_print_runtime_diagnostics"
    )
    calls = [
        node
        for node in ast.walk(diagnostics)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_verify_shovel_hand_actuator_runtime"
    ]
    assert len(calls) == 1
    assert "shovel_configuration" in ast.unparse(diagnostics)
