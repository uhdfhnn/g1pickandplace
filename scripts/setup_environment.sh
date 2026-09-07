#!/usr/bin/env bash
set -euo pipefail

# This bootstrap reproduces the public stack used by the accepted visible runs.
# It intentionally installs into sibling checkouts because Unitree's upstream
# auto_setup_env.sh assumes that layout. Run it after cloning g1pickandplace.

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname "$REPOSITORY_ROOT")"

# Isaac Sim 5.0 is the validated major/minor for the RTX 5090 host and Unitree's
# documented RTX 50-series path. The value is a software release, not a physical
# calibration. 4.5 can lack RTX 50 support and 5.1 has unvalidated API/runtime
# drift, so changing it requires a fresh install plus every visible gate. It is
# fixed here while the lower-level Unitree script remains multi-version.
readonly ISAAC_SIM_SERIES="5.0"

# Python 3.11 and the cu126 wheel channel are selected by Unitree's public Isaac
# Sim 5.0 installer. They have no coordinate frame or units beyond the CUDA
# toolkit release encoded by cu126. Older Python/CUDA builds can miss required
# wheels; newer ones may make Isaac Sim or Torch resolve differently. These are
# fixed bootstrap inputs and must be revalidated together if changed upstream.
readonly CUDA_WHEEL_CHANNEL="cu126"

# This Conda name matches Unitree's public examples and the accepted-run host.
# It is only a host-local identifier, so callers may override it without changing
# robot behavior. The selected environment must still pass check_install.py;
# pointing to an existing environment is rejected to avoid silently mixing stacks.
readonly CONDA_ENVIRONMENT="${G1PICKPLACE_CONDA_ENV:-unitree_sim_env}"

# Repository commits are immutable provenance pins from the accepted runs. They
# identify source snapshots rather than numeric controller values. A commit that
# is older may lack the exercised task/assets; a newer one may change task
# registration, joint ordering, DDS, geometry, or ABI behavior. Overrides are
# intentionally not supported: update these values only with a documented clean
# setup, visible inspect/plan/rollout evidence, and the full test suite.
readonly UNITREE_SIM_COMMIT="e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc"
readonly UNITREE_ROS_COMMIT="7d6075f7f58588b189b940130e3edab3c839b2df"
readonly ISAACLAB_COMMIT="46dff135f44683f031edf346e544fcfd8456b2bb"
readonly UNITREE_SDK_COMMIT="65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5"
readonly CYCLONEDDS_COMMIT="5041f3560c088c99e5088b2b8520b69169621196"
readonly TELEIMAGER_COMMIT="b81de448bca9c696d7ce145f4af71c66146d0b69"

readonly UNITREE_SIM_DIR="$WORKSPACE_ROOT/unitree_sim_isaaclab"
readonly UNITREE_ROS_DIR="$WORKSPACE_ROOT/unitree_ros"
readonly ISAACLAB_DIR="$WORKSPACE_ROOT/IsaacLab"
readonly UNITREE_SDK_DIR="$WORKSPACE_ROOT/unitree_sdk2_python"
readonly CYCLONEDDS_DIR="$WORKSPACE_ROOT/cyclonedds"

if ! command -v git >/dev/null 2>&1; then
    echo "error: git is required" >&2
    exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda is required; install Miniconda/Anaconda and place conda on PATH" >&2
    exit 1
fi

clone_or_pin() {
    local repository_url="$1"
    local destination="$2"
    local commit="$3"

    if [[ ! -e "$destination" ]]; then
        git clone "$repository_url" "$destination"
    elif [[ ! -d "$destination/.git" ]]; then
        echo "error: existing path is not a Git checkout: $destination" >&2
        exit 1
    fi

    if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]]; then
        echo "error: refusing to switch a modified dependency checkout: $destination" >&2
        exit 1
    fi

    if ! git -C "$destination" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        git -C "$destination" fetch origin "$commit"
    fi
    git -C "$destination" checkout --detach "$commit"
}

clone_or_pin \
    "https://github.com/unitreerobotics/unitree_sim_isaaclab.git" \
    "$UNITREE_SIM_DIR" \
    "$UNITREE_SIM_COMMIT"
clone_or_pin \
    "https://github.com/unitreerobotics/unitree_ros.git" \
    "$UNITREE_ROS_DIR" \
    "$UNITREE_ROS_COMMIT"
clone_or_pin \
    "https://github.com/isaac-sim/IsaacLab.git" \
    "$ISAACLAB_DIR" \
    "$ISAACLAB_COMMIT"
clone_or_pin \
    "https://github.com/unitreerobotics/unitree_sdk2_python.git" \
    "$UNITREE_SDK_DIR" \
    "$UNITREE_SDK_COMMIT"
clone_or_pin \
    "https://github.com/eclipse-cyclonedds/cyclonedds.git" \
    "$CYCLONEDDS_DIR" \
    "$CYCLONEDDS_COMMIT"

git -C "$UNITREE_SIM_DIR" submodule update --init --recursive
actual_teleimager_commit="$(git -C "$UNITREE_SIM_DIR/teleimager" rev-parse HEAD)"
if [[ "$actual_teleimager_commit" != "$TELEIMAGER_COMMIT" ]]; then
    echo "error: teleimager commit mismatch: $actual_teleimager_commit" >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENVIRONMENT"; then
    echo "error: Conda environment already exists: $CONDA_ENVIRONMENT" >&2
    echo "Use a new G1PICKPLACE_CONDA_ENV value or remove the old environment explicitly." >&2
    exit 1
fi

(
    cd "$UNITREE_SIM_DIR"
    bash auto_setup_env.sh \
        "$ISAAC_SIM_SERIES" \
        "$CONDA_ENVIRONMENT" \
        "$CUDA_WHEEL_CHANNEL"
)

# Install the repository plus the exact Pinocchio version declared in
# pyproject.toml. Keeping this after Unitree's installer lets the project pin be
# the final resolver decision instead of being overwritten by upstream setup.
conda run --no-capture-output -n "$CONDA_ENVIRONMENT" \
    python -m pip install -e "${REPOSITORY_ROOT}[dev,isaac]"

# Only LeRobot's native dataset subsystem is used. Its full distribution declares
# policy/training packages that are unrelated to recording and conflict with the
# Isaac Sim prebundle. The requirements file captures every dataset dependency
# from the accepted environment; --no-deps prevents pip from widening that exact
# set. check_install.py immediately verifies both versions and the real import.
conda run --no-capture-output -n "$CONDA_ENVIRONMENT" \
    python -m pip install --no-deps -r "$REPOSITORY_ROOT/requirements-recording.txt"

UNITREE_SIM_ISAACLAB_ROOT="$UNITREE_SIM_DIR" \
    conda run --no-capture-output -n "$CONDA_ENVIRONMENT" \
    python "$REPOSITORY_ROOT/scripts/check_install.py"

conda run --no-capture-output -n "$CONDA_ENVIRONMENT" \
    python -m pytest -q "$REPOSITORY_ROOT/tests"

echo
echo "Setup complete."
echo "Activate with: conda activate $CONDA_ENVIRONMENT"
echo "Run from:      $REPOSITORY_ROOT"
