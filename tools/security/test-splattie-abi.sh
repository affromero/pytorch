#!/usr/bin/env bash
set -euo pipefail

wheel_path=$(realpath "$1")
wheel_sha=$2
source_revision=$3
printf '%s  %s\n' "$wheel_sha" "$wheel_path" | sha256sum --check
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl libgomp1 libgl1 libglib2.0-0 libfribidi0
rm -rf /var/lib/apt/lists/*
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
export UV_LINK_MODE=copy
mkdir -p /abi-check
cd /abi-check
cat > pyproject.toml <<TOML
[project]
name = "splattie-native-abi-check"
version = "0.0.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "torch==2.3.0", "torchvision==0.18.0", "xformers==0.0.26.post1", "numpy==1.26.4",
  "torch-scatter @ https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp311-cp311-linux_x86_64.whl",
  "kaolin @ https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.3.0_cu121/kaolin-0.18.0-cp311-cp311-linux_x86_64.whl",
  "flash-attn @ https://github.com/affromero/flash-attention/releases/download/security-2.5.9.post1-1/flash_attn-2.5.9.post1%2Bsecurity.1-cp311-cp311-linux_x86_64.whl",
  "pillow @ https://github.com/affromero/Pillow/releases/download/security-11.3.0-1/pillow-11.3.0%2Bsecurity.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
]
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
[tool.uv.sources]
torch = { path = "$wheel_path" }
torchvision = { index = "pytorch-cu121" }
xformers = { index = "pytorch-cu121" }
TOML
uv sync --python 3.11.15
.venv/bin/python - "$source_revision" <<'PY'
import importlib.metadata
from pathlib import Path
import sys

import torch
import torchvision
import torch_scatter
import flash_attn
import flash_attn_2_cuda
import kaolin
import kaolin._C
import xformers
import xformers.ops

assert torch.__version__ == "2.3.0+cu121.security1", torch.__version__
assert torch.version.git_version == sys.argv[1], torch.version.git_version
assert torch.version.cuda == "12.1", torch.version.cuda
assert not torch._C._GLIBCXX_USE_CXX11_ABI
assert torch.cuda.device_count() == 0
assert torch.backends.cudnn.version() == 8902

boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]])
kept = torchvision.ops.nms(boxes, torch.tensor([.9, .8, .7]), .5)
torch.testing.assert_close(kept, torch.tensor([0, 2]))
scattered = torch_scatter.scatter_add(torch.tensor([1., 2., 3., 4.]), torch.tensor([0, 1, 0, 1]))
torch.testing.assert_close(scattered, torch.tensor([4., 6.]))
maxima, argmax = torch_scatter.scatter_max(torch.tensor([1., 2., 3., 4.]), torch.tensor([0, 1, 0, 1]))
torch.testing.assert_close(maxima, torch.tensor([3., 4.]))
torch.testing.assert_close(argmax, torch.tensor([2, 3]))
vertices = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
faces = torch.tensor([[0, 1, 2], [1, 2, 3]])
indexed = kaolin.ops.mesh.index_vertices_by_faces(vertices, faces)
torch.testing.assert_close(indexed, torch.stack([vertices[:, :3], vertices[:, 1:]], dim=1))
tetrahedron = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
closed_faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
points = torch.tensor([[[.1, .1, .1], [2., 2., 2.]]])
torch.testing.assert_close(kaolin.ops.mesh.check_sign(tetrahedron, closed_faces, points), torch.tensor([[True, False]]))
native_xformers = Path(xformers.__file__).parent / "_C.so"
assert native_xformers.is_file(), native_xformers
torch.ops.load_library(str(native_xformers))
for name in ("torch", "torchvision", "torch-scatter", "kaolin", "flash-attn", "xformers", "pillow"):
    print(f"{name}={importlib.metadata.version(name)}")
print("CPU NMS, native scatter maxima and native mesh containment passed. All native extensions loaded.")
print("GPU kernels and generation were not executed.")
PY
