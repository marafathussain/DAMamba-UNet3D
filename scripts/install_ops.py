#!/usr/bin/env python3
"""
Download and build the DAMamba CUDA extensions (DCNv3 + selective_scan_cuda_oflex_rh)
from a PINNED upstream commit, then verify that both are importable.

The pin makes builds reproducible: every reviewer who runs this script gets the
exact same sources, regardless of when ltzovo/DAMamba advances main.

Run on a GPU node (not the login node), after activating the damamba_unet env:

    source /home/ch225256/miniconda3/etc/profile.d/conda.sh
    conda activate damamba_unet
    python scripts/install_damamba_ops.py

What this writes:
    third_party/damamba/ops_dcnv3/...          (source + built extension)
    third_party/damamba/selective_scan/...     (source + built extension)
    third_party/damamba/build_env.json         (versions, arch list, commit SHA)

The script will:
    - set TORCH_CUDA_ARCH_LIST automatically if the env var is not already set
      (defaults to a wide multi-arch list covering V100/T4/A100/A40/3090/L40/H100)
    - hard-fail if either `import DCNv3` or `import selective_scan_cuda_oflex_rh`
      fails after the build (no silent success)
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

UPSTREAM_REPO = "ltzovo/DAMamba"
PINNED_COMMIT = "dc66afb584c704bdb0b89cad0eb0c4ddd36c193d"
TREE_API = f"https://api.github.com/repos/{UPSTREAM_REPO}/git/trees/{PINNED_COMMIT}?recursive=1"
RAW_BASE = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{PINNED_COMMIT}/"

DEFAULT_ARCH_LIST = "7.0;7.5;8.0;8.6;8.9;9.0"


def fetch_tree() -> list[dict]:
    with urllib.request.urlopen(TREE_API) as resp:
        payload = json.load(resp)
    if payload.get("truncated"):
        raise RuntimeError(
            "GitHub returned a truncated tree for the pinned DAMamba commit. "
            "This is unexpected for such a small repo."
        )
    return payload["tree"]


def download_file(path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = RAW_BASE + path
    with urllib.request.urlopen(url) as resp:
        dest.write_bytes(resp.read())


def download_dir(prefix: str, target_root: Path) -> int:
    tree = fetch_tree()
    files = [item["path"] for item in tree if item["type"] == "blob" and item["path"].startswith(prefix)]
    if not files:
        raise RuntimeError(f"No files found for prefix {prefix} at {PINNED_COMMIT}")
    target_root.mkdir(parents=True, exist_ok=True)
    for path in files:
        rel = Path(path).relative_to(prefix)
        download_file(path, target_root / rel)
    return len(files)


def detect_arch_list() -> str:
    """Honor user override; otherwise auto-detect the local GPU; otherwise multi-arch."""
    env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env:
        return env
    try:
        import torch  # noqa: WPS433 (local import: torch must exist by now)
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return f"{cap[0]}.{cap[1]}"
    except Exception:
        pass
    return DEFAULT_ARCH_LIST


def run(cmd: list[str], cwd: Path, env: dict) -> None:
    print(f"+ ({cwd.name}) {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=cwd, env=env)


def torch_lib_path() -> str:
    import torch

    return os.path.join(os.path.dirname(torch.__file__), "lib")


def env_with_torch_libs(base: dict | None = None) -> dict:
    """DCNv3 / selective_scan link against libtorch; ensure libc10.so is on LD_LIBRARY_PATH."""
    env = dict(base or os.environ)
    torch_lib = torch_lib_path()
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{prev}" if prev else torch_lib
    return env


def assert_importable(module_name: str, hint: str, env: dict) -> str:
    """Import in a SUBPROCESS to bypass any caching from earlier import attempts."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {module_name}; print(getattr({module_name}, '__file__', '<built-in>'))",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(
            f"Built {module_name} but it is not importable. {hint}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def gather_build_env(arch_list: str) -> dict:
    info: dict = {
        "upstream_repo": UPSTREAM_REPO,
        "pinned_commit": PINNED_COMMIT,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch_cuda_arch_list": arch_list,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        info["torch"] = "unavailable"
    for tool in (("nvcc", ["nvcc", "--version"]), ("gcc", ["gcc", "--version"])):
        try:
            out = subprocess.check_output(tool[1], stderr=subprocess.STDOUT, text=True)
            info[tool[0]] = out.splitlines()[0]
        except Exception:
            info[tool[0]] = "unavailable"
    return info


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    third_party = repo_root / "third_party" / "damamba"
    third_party.mkdir(parents=True, exist_ok=True)

    dcn_dir = third_party / "ops_dcnv3"
    sel_dir = third_party / "selective_scan"

    arch_list = detect_arch_list()
    build_env_info = gather_build_env(arch_list)

    print(f"=== DAMamba CUDA extension installer ===")
    print(f"  pinned commit:        {PINNED_COMMIT}")
    print(f"  TORCH_CUDA_ARCH_LIST: {arch_list}")
    print(f"  torch:                {build_env_info.get('torch')}")
    print(f"  torch CUDA build:     {build_env_info.get('torch_cuda_build')}")
    print(f"  nvcc:                 {build_env_info.get('nvcc')}")

    print("\n[1/4] Downloading DCNv3 sources at pinned commit ...", flush=True)
    n_dcn = download_dir("classification/models/ops_dcnv3/", dcn_dir)
    print(f"  downloaded {n_dcn} files")

    print("\n[2/4] Downloading selective_scan sources at pinned commit ...", flush=True)
    n_sel = download_dir("classification/models/selective_scan/", sel_dir)
    print(f"  downloaded {n_sel} files")

    env = env_with_torch_libs()
    env["TORCH_CUDA_ARCH_LIST"] = arch_list

    print("\n[3/4] Building DCNv3 extension ...", flush=True)
    run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", "."], dcn_dir, env)

    print("\n[4/4] Building selective_scan extension ...", flush=True)
    run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", "."], sel_dir, env)

    print("\n=== Post-build import check ===")
    dcn_path = assert_importable(
        "DCNv3",
        f"This usually means TORCH_CUDA_ARCH_LIST ({arch_list}) does not match the GPU "
        "you build on, or nvcc/torch CUDA mismatch.",
        env,
    )
    print(f"  DCNv3                            -> {dcn_path}")
    ss_path = assert_importable(
        "selective_scan_cuda_oflex_rh",
        "Same hint as DCNv3: check CUDA toolkit and arch list.",
        env,
    )
    print(f"  selective_scan_cuda_oflex_rh     -> {ss_path}")

    build_env_info["dcnv3_path"] = dcn_path
    build_env_info["selective_scan_path"] = ss_path
    (third_party / "build_env.json").write_text(json.dumps(build_env_info, indent=2))
    print(f"\nBuild env recorded at: {third_party / 'build_env.json'}")
    print("DONE. Now run: python scripts/verify.py")


if __name__ == "__main__":
    main()
