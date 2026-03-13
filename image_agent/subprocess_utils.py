from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class CmdResult:
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str


def _which(program: str) -> Optional[str]:
    return shutil.which(program)


def _find_conda_exe() -> str:
    """
    Return the path to a conda executable.

    Prefer PATH (shutil.which), but fall back to $CONDA_EXE if PATH isn't set up.
    This fixes cases where 'conda' is available via shell init but not on PATH,
    while $CONDA_EXE points to the real executable.
    """
    exe = _which("conda")
    if exe:
        return exe

    exe = os.environ.get("CONDA_EXE")
    if exe and os.path.isfile(exe) and os.access(exe, os.X_OK):
        return exe

    raise RuntimeError(
        "conda not found on PATH and $CONDA_EXE is not a usable executable. "
        "Fix PATH (e.g., export PATH=.../bin:$PATH) or set CONDA_EXE."
    )


def build_env_prefix(env_name: Optional[str], runner: str = "conda") -> List[str]:
    """
    Returns a prefix list to run a command in a named environment.

    Default assumes conda environments:
      conda run -n <env_name> <cmd...>

    If env_name is None, returns [].
    """
    if not env_name:
        return []

    runner = runner.lower().strip()

    if runner == "conda":
        # ✅ Use the robust resolver (PATH first, fallback to $CONDA_EXE)
        exe = _find_conda_exe()
        return [exe, "run", "-n", env_name]

    if runner == "mamba":
        exe = _which("mamba")
        if not exe:
            raise RuntimeError("mamba not found on PATH, cannot run env command.")
        return [exe, "run", "-n", env_name]

    raise ValueError(f"Unknown runner: {runner}")


def run_cmd(
    cmd: List[str],
    env_name: Optional[str] = None,
    env_runner: str = "conda",
    cwd: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    timeout_s: Optional[int] = None,
) -> CmdResult:
    prefix = build_env_prefix(env_name, runner=env_runner)
    full_cmd = prefix + cmd

    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    proc = subprocess.run(
        full_cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    return CmdResult(
        cmd=full_cmd,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def extract_first_json(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from a string that may contain extra lines.
    Useful for scripts that print JSON then additional logs.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in text (missing '{').")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return json.loads(candidate)

    raise ValueError("Unterminated JSON object in text.")
