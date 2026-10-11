#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates libgomp1 binutils
rm -rf /var/lib/apt/lists/*
mkdir /wheel-test
cd /wheel-test
wheel_path=$(realpath /wheelhouse/torch-*.whl)
cat > pyproject.toml <<TOML
[project]
name = "torch-relocation-test"
version = "0.0.0"
requires-python = ">=3.11,<3.12"
dependencies = ["torch", "numpy==1.26.4"]
[tool.uv.sources]
torch = { path = "$wheel_path" }
TOML
uv sync --python 3.11.15
.venv/bin/python - <<'PY'
import pathlib
import subprocess
import torch
assert torch.__version__ == "2.3.0+cu121.security1", torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert torch.backends.cudnn.version() == 8902
assert torch.backends.mkldnn.is_available()
assert not torch._C._GLIBCXX_USE_CXX11_ABI
assert torch.cuda.device_count() == 0
for library in (pathlib.Path(torch.__file__).parent / "lib").glob("*.so*"):
    result = subprocess.run(["readelf", "-d", str(library)], check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "RPATH" in line or "RUNPATH" in line:
            print(library.name, line)
            assert "/tmp/torch-security-build" not in line, line
print(torch.__config__.show())
PY
.venv/bin/python /regressions.py -v
