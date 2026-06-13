"""Run/task setup: load task reference files and create the run directory.

Hosts the TaskFiles/RunSetup containers and the filesystem-prep helpers previously
defined in orchestrator.py (re-exported there for the existing test contract).
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sia.config import Config
from sia.context_manager import ContextManager
from sia.layout import RunLayout, TaskLayout, venv_pip_path, venv_python_path
from sia.logging_setup import get_logger

if TYPE_CHECKING:
    from sia.agent_reference import ResolvedAgentReference
    from sia.profiles import MetaAgentProfile, TargetAgentProfile

logger = get_logger(__name__)


@dataclass
class TaskFiles:
    """Container for task reference files loaded from disk."""

    sample_task_descriptions: str
    reference_target_agent_py: str
    sample_agent_execution: dict
    task_md: str


@dataclass
class RunSetup:
    """Container for run directory paths and managers."""

    run_directory: str
    meta_agent_working_directory: str
    venv_dir: str
    context_mgr: ContextManager


def load_task_files(
    task_dir: str,
    shared_dir: str,
    resolved_ref: ResolvedAgentReference | None = None,
) -> TaskFiles:
    """Load all reference files from the task directory.

    The seed shown to the meta-agent comes from ``resolved_ref`` (the target profile's
    agent_reference) when provided: its ``inline_seed`` for a default/single-file
    reference, or empty for a multi-file directory reference (the agent reads that from
    disk). When ``resolved_ref`` is None, fall back to the task's bundled reference.
    """
    logger.info("Loading files from task directory...")
    paths = TaskLayout(task_dir, shared_dir)

    sample_task_descriptions = Path(paths.sample_descriptions).read_text()
    logger.info("  ✓ Sample task descriptions loaded")

    if resolved_ref is None:
        reference_target_agent_py = Path(paths.reference_agent).read_text()
    else:
        reference_target_agent_py = resolved_ref.inline_seed or ""
    logger.info("  ✓ Reference target agent loaded")

    with open(paths.sample_execution) as f:
        sample_agent_execution = json.load(f)
    logger.info("  ✓ Sample agent execution loaded")

    task_md = Path(paths.task_md).read_text()
    logger.info("  ✓ Task specification loaded")

    return TaskFiles(
        sample_task_descriptions=sample_task_descriptions,
        reference_target_agent_py=reference_target_agent_py,
        sample_agent_execution=sample_agent_execution,
        task_md=task_md,
    )


def _create_venv(venv_dir: str, packages: list[str]) -> None:
    """Create a virtual environment and install packages."""
    if shutil.which("uv"):
        subprocess.run(["uv", "venv", venv_dir], check=True)
        subprocess.run(
            ["uv", "pip", "install", "--python", venv_python_path(venv_dir), *packages],
            check=True,
        )
    else:
        venv.create(venv_dir, with_pip=True)
        subprocess.run([venv_pip_path(venv_dir), "install", *packages], check=True)


def install_requirements(venv_dir: str, requirements_path: str) -> None:
    """Install a requirements.txt into an existing venv (augmenting the baseline packages).

    Used per generation so the meta/feedback agents can evolve the target's
    dependencies by editing requirements.txt across generations.
    """
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", venv_python_path(venv_dir), "-r", requirements_path]
    else:
        cmd = [venv_pip_path(venv_dir), "install", "-r", requirements_path]
    logger.info(f"Installing generation dependencies from {requirements_path}")
    subprocess.run(cmd, check=True)


def _write_run_profiles(
    run_directory: str,
    meta_profile: MetaAgentProfile | None,
    target_profile: TargetAgentProfile | None,
) -> None:
    """Persist the resolved meta/target profiles as ``profiles.json`` in the run dir.

    Dumping the whole profile object (provider details + resolved agent_reference,
    whose ``source`` is already an absolute path) means the web visualizer renders
    full profile detail generically — no per-field plumbing, and new profile fields
    show up automatically.
    """
    profiles = {
        role: dataclasses.asdict(profile)
        for role, profile in (("meta", meta_profile), ("target", target_profile))
        if profile is not None
    }
    if not profiles:
        return
    path = os.path.join(run_directory, "profiles.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profiles, fh, indent=2, default=str)  # default=str: Path -> string


def setup_tpu_environment() -> None:
    """Configure environment variables for TPU v5e-8 training.

    Sets PJRT (hardware-agnostic TPU runtime) and libtpu paths.
    Call this before any TPU operations (importing torch_xla, etc.).
    """
    logger.info("Setting up TPU v5e environment...")

    # PJRT is the default TPU runtime for v5e chips
    os.environ.setdefault("PJRT_DEVICE", "TPU")

    # XLA flags for TPU compilation
    os.environ.setdefault("XLA_USE_BF16", "1")
    os.environ.setdefault("XLA_USE_XRT", "1")

    # TPU pod configuration (v5e-8 = 4 chips on 1 host slice)
    os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "4,1,1")
    os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")

    # Memory and compilation settings for v5e
    os.environ.setdefault("TPU_MEGACORE", "1")
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")

    # Log TPU device info
    logger.info("  ✓ PJRT_DEVICE=%s", os.environ.get("PJRT_DEVICE"))
    logger.info("  ✓ XLA_USE_BF16=%s", os.environ.get("XLA_USE_BF16"))

    # Attempt to detect TPUs
    try:
        import subprocess
        result = subprocess.run(
            ["ls", "/dev/accel*", "/dev/vfio*", "/dev/tpu*"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout:
            logger.info("  ✓ TPU devices found: %s", result.stdout.strip().replace("\n", ", "))
        else:
            logger.warning("  ⚠ No TPU devices found in /dev/. Is the TPU runtime loaded?")
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("  ⚠ Could not probe TPU devices: %s", e)


def check_tpu_availability() -> bool:
    """Check if TPU v5e devices are reachable by importing torch-xla.

    Returns True if at least one TPU core is available, False otherwise.
    """
    try:
        import torch_xla.core.xla_model as xm
        device_count = xm.xla_device_count()
        if device_count == 0:
            logger.warning("  ⚠ torch_xla found but 0 TPU devices detected.")
            return False
        logger.info("  ✓ %d TPU device(s) detected via torch_xla", device_count)
        return True
    except ImportError:
        logger.warning("  ⚠ torch_xla not installed. TPU training unavailable.")
        return False
    except Exception as e:
        logger.warning("  ⚠ TPU check failed: %s", e)
        return False


def setup_run_directory(
    run_id: int,
    task_dir: str,
    meta_model: str,
    task_model: str,
    agent_impl: str,
    max_gen: int,
    focus: str = "harness",
    training_backend: str = "tinker-api",
    config: Config | None = None,
    meta_profile: MetaAgentProfile | None = None,
    target_profile: TargetAgentProfile | None = None,
) -> RunSetup:
    """Create run directories, venv, and context manager.

    Args:
        focus: "harness" (default) or "weights" for RL-based tuning. Determines which packages are installed.
        training_backend: "tinker-api" (default) or "tpu-native". Only used when focus="weights".
    """
    cfg = config or Config()
    layout = RunLayout.for_run_id(run_id)
    run_directory = layout.run_dir
    meta_agent_working_directory = layout.gen_dir(1)

    if os.path.exists(run_directory):
        logger.error(f"Run directory already exists: {run_directory}")
        logger.error("Please use a different run_id or remove the existing directory")
        sys.exit(1)

    logger.info(f"Creating run directory: {run_directory}")
    os.makedirs(run_directory, exist_ok=False)

    logger.info(f"Creating meta_agent working directory: {meta_agent_working_directory}")
    os.makedirs(meta_agent_working_directory, exist_ok=False)

    venv_dir = layout.venv_dir
    logger.info(f"Creating virtual environment at: {venv_dir}")
    # Build packages list conditionally based on focus mode and training backend
    packages = cfg.VENV_PACKAGES.copy()
    if focus == "weights":
        packages.extend(cfg.WEIGHTS_VENV_PACKAGES)
        if training_backend == "tpu-native":
            packages = [p for p in packages if "tinker" not in p and "vllm" not in p]
            packages.extend(cfg.TPU_VENV_PACKAGES)
            setup_tpu_environment()
    _create_venv(venv_dir, packages)

    _write_run_profiles(run_directory, meta_profile, target_profile)

    logger.info("Initializing context manager...")
    context_mgr = ContextManager(
        run_directory,
        {
            "task_dir": task_dir,
            "meta_model": meta_model,
            "task_model": task_model,
            "agent_impl": agent_impl,
            "max_gen": max_gen,
        },
    )
    context_mgr.initialize()
    logger.info("  ✓ Context manager initialized")

    return RunSetup(
        run_directory=run_directory,
        meta_agent_working_directory=meta_agent_working_directory,
        venv_dir=venv_dir,
        context_mgr=context_mgr,
    )
