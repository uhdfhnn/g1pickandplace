from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"


def _load_frame_helpers():
    """Load the dependency-light filename/writer helpers without Isaac Sim."""

    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    wanted = {
        "_phase_boundary_frame_filename",
        "_save_phase_boundary_frames",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    namespace = {
        "Mapping": dict,
        "Path": Path,
        "np": np,
        "PHASE_BOUNDARY_CAMERA_NAMES": ("front", "right_wrist"),
        # These widths are formatting choices copied from the runner; keeping
        # them explicit lets this test execute without importing its parser or
        # simulator dependencies.
        "PHASE_BOUNDARY_FILENAME_INDEX_WIDTH": 3,
        "PHASE_BOUNDARY_FILENAME_STEP_WIDTH": 6,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace["_phase_boundary_frame_filename"], namespace["_save_phase_boundary_frames"]


def test_phase_boundary_filename_is_sorted_and_sanitized() -> None:
    filename, _ = _load_frame_helpers()
    assert filename(2, "move/to grasp", 41, "front") == (
        "boundary-002-move_to_grasp-step-000041-front.png"
    )
    assert filename(10, "return_home", 1200, "right_wrist") == (
        "boundary-010-return_home-step-001200-right_wrist.png"
    )


def test_phase_boundary_writer_saves_only_required_rgb_cameras(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    _, save_frames = _load_frame_helpers()
    front = np.zeros((3, 4, 3), dtype=np.uint8)
    front[1, 2] = (12, 34, 56)
    right_wrist = np.full((2, 5, 4), 200, dtype=np.uint8)
    save_frames(
        tmp_path,
        boundary_index=0,
        phase="open_at_home",
        step=0,
        images={"front": front, "right_wrist": right_wrist, "other": front},
    )

    files = sorted(path.name for path in tmp_path.iterdir())
    assert files == [
        "boundary-000-open_at_home-step-000000-front.png",
        "boundary-000-open_at_home-step-000000-right_wrist.png",
    ]

    from PIL import Image

    with Image.open(tmp_path / files[0]) as saved_front:
        assert saved_front.mode == "RGB"
        assert saved_front.size == (4, 3)
        assert saved_front.getpixel((2, 1)) == (12, 34, 56)
    with Image.open(tmp_path / files[1]) as saved_wrist:
        assert saved_wrist.mode == "RGB"
        assert saved_wrist.size == (5, 2)


def test_phase_boundary_writer_skips_missing_camera_without_error(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    _, save_frames = _load_frame_helpers()
    save_frames(
        tmp_path,
        boundary_index=1,
        phase="descend_to_grasp",
        step=7,
        images={"front": np.zeros((2, 2, 3), dtype=np.uint8)},
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "boundary-001-descend_to_grasp-step-000007-front.png"
    ]
