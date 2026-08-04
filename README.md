# DAMamba-UNet3D

PyTorch implementation of **DAMamba-UNet3D**, a hybrid 3D U-Net with **Dynamic Adaptive Scan (DAS)** extended to tri-plane volumetric selective scan. DAS learns input-dependent voxel ordering before Mamba state-space mixing; convolutions handle local detail elsewhere.

Paper: [DAMamba-UNet3D (arXiv)](https://arxiv.org/abs/2607.22718)

## Models

| Model | Role | ~Params |
|-------|------|---------|
| `DAMambaUNet3D` | Compact encoder-only DAS (E2–E4) | ~5.3M |
| `DAMambaUNet3DLarge` | Wide DAS-native U-Net (DAMamba-L) | ~70M |

Both require the DAMamba CUDA extensions (DCNv3 + selective scan).

## Requirements

- Linux, Python ≥ 3.10
- NVIDIA GPU with CUDA (extensions are built from source)
- PyTorch with matching CUDA toolkit
- `nvcc`, `gcc`, and a C++ compiler for extension builds

## Install

```bash
git clone https://github.com/marafathussain/DAMamba-UNet3D.git
cd DAMamba-UNet3D

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .

# Build DCNv3 + selective_scan (run on a machine with GPU + nvcc)
python scripts/install_ops.py
```

`install_ops.py` downloads pinned sources from [ltzovo/DAMamba](https://github.com/ltzovo/DAMamba) and installs them under `third_party/damamba/`.

Set `TORCH_CUDA_ARCH_LIST` if you build on a CPU node for a specific GPU (e.g. `export TORCH_CUDA_ARCH_LIST=8.6` for A40).

## Verify

```bash
python scripts/verify.py
```

Checks extension imports, forward pass, and backward pass for both model sizes.

## Quick demo

```bash
# Compact model (~5.3M), random 128^3 volume
python scripts/demo.py --model compact

# Large model (~70M); use a smaller volume if GPU memory is tight
python scripts/demo.py --model large --depth 64 --height 64 --width 64
```

## Use in your own project

```python
import torch
from damamba_unet3d import DAMambaUNet3D

model = DAMambaUNet3D(
    in_channels=4,
    out_channels=4,
    base_width=16,
    use_damamba_in=["E2", "E3", "E4"],
    bottleneck_depth=1,
    damamba_cfg={"d_state": 16, "d_conv": 4, "expand": 2, "scan_mode": "das"},
).cuda()

x = torch.randn(1, 4, 128, 128, 128, device="cuda")
y = model(x)  # (1, 4, 128, 128, 128)
```

See `configs/default.yaml` for a full hyperparameter template. This repository provides the **architecture and CUDA backend**; plug it into your own dataset loader and training loop.

## Repository layout

```
damamba_unet3d/          # Python package
  models/                 # U-Net, DASSM3D, tri-plane DAS
  ops/selective_scan/     # Python wrapper for CUDA selective scan
scripts/
  install_ops.py          # Build DAMamba CUDA extensions
  verify.py               # GPU smoke test
  demo.py                 # Forward-pass example
configs/default.yaml      # Model hyperparameters
CITATION.bib              # BibTeX for this work
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{damamba_unet3d2026,
  title   = {{DAMamba-UNet3D}: A Parameter-Efficient Mamba State Space {U-Net} with Dynamic Adaptive Scan for {3D} Medical Image Segmentation},
  author  = {Hussain, Mohammad Arafat and Grant, Ellen and Ou, Yangming},
  journal = {arXiv preprint arXiv:2607.22718},
  year    = {2026}
}
```

Also cite the upstream DAS and Mamba work:

- DAMamba (2D DAS): Liu et al., CVPR 2025
- Mamba: Gu & Dao, arXiv:2312.00752

## Acknowledgements

CUDA operators are adapted from [ltzovo/DAMamba](https://github.com/ltzovo/DAMamba) (DCNv3 adaptive scan and selective scan kernels).

## License

MIT — see [LICENSE](LICENSE).
