#!/usr/bin/env bash
set -euo pipefail

test "$(uname -s)" = Darwin
test "$(uname -m)" = arm64
source_root=$PWD
source_revision=$(git rev-parse HEAD)
test "$source_revision" = "$EXPECTED_SOURCE_REVISION"
build_env="$RUNNER_TEMP/torch-macos-build"
mkdir -p "$build_env"
cat > "$build_env/pyproject.toml" <<'TOML'
[project]
name = "torch-macos-security-build"
version = "0.0.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "setuptools==83.0.0", "wheel==0.46.2", "cmake==3.29.6", "ninja==1.11.1.1",
  "numpy==1.26.4", "astunparse==1.6.3", "pyyaml==6.0.2", "packaging==24.2",
  "typing-extensions>=4.8", "requests", "delocate==0.13.0",
]
TOML
uv sync --project "$build_env" --python 3.11.15
export PATH="$build_env/.venv/bin:$PATH"
export VIRTUAL_ENV="$build_env/.venv"
export MAX_JOBS=2 USE_CUDA=0 USE_MPS=1 BUILD_TEST=0 BUILD_CAFFE2=0
export CMAKE_BUILD_TYPE=Release CMAKE_OSX_ARCHITECTURES=arm64
export MACOSX_DEPLOYMENT_TARGET=11.0
openmp_prefix=$(brew --prefix libomp)
export CMAKE_PREFIX_PATH="$openmp_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
unset PYTORCH_BUILD_VERSION PYTORCH_BUILD_NUMBER PYTORCH_EXTRA_INSTALL_REQUIREMENTS
python test/test_security_packaging.py
python setup.py bdist_wheel

# Bundled libraries determine the actual minimum supported macOS version.
env -u MACOSX_DEPLOYMENT_TARGET delocate-wheel --require-archs arm64 --wheel-dir repaired dist/*.whl
rm dist/*.whl
mv repaired/*.whl dist/
cp "$build_env/uv.lock" dist/build-tools-uv.lock
{
  printf '%s\n' "$source_revision"
  sw_vers
  xcodebuild -version
  xcrun --show-sdk-version
  clang --version
  brew list --versions libomp
  "$build_env/.venv/bin/python" --version
} > dist/build-provenance.txt
shasum -a 256 dist/*.whl > dist/SHA256SUMS
wheel_path=$(python -c 'import pathlib; print(next(pathlib.Path("dist").glob("*.whl")).resolve())')
test_env="$RUNNER_TEMP/torch-macos-runtime"
mkdir -p "$test_env"
cat > "$test_env/pyproject.toml" <<TOML
[project]
name = "torch-macos-security-runtime"
version = "0.0.0"
requires-python = ">=3.11,<3.12"
dependencies = ["torch", "numpy==1.26.4", "setuptools==84.0.0"]
[tool.uv.sources]
torch = { path = "$wheel_path" }
TOML
uv sync --project "$test_env" --python 3.11.15
cd "$test_env"
.venv/bin/python - <<'PY'
import os
import pathlib
import subprocess

import torch

assert torch.__version__ == "2.3.0+security.1", torch.__version__
assert torch.version.git_version == os.environ["EXPECTED_SOURCE_REVISION"], torch.version.git_version
assert torch.version.cuda is None, torch.version.cuda
assert torch.backends.mps.is_built(), "The maintained Mac build must retain MPS support"
root = pathlib.Path(torch.__file__).parent
for library in root.rglob("*"):
    if library.suffix not in {".so", ".dylib"}:
        continue
    architectures = subprocess.check_output(["lipo", "-archs", str(library)], text=True).strip()
    assert architectures == "arm64", (library, architectures)
    dependencies = subprocess.check_output(["otool", "-L", str(library)], text=True)
    print(dependencies)
    for line in dependencies.splitlines()[1:]:
        assert "/opt/homebrew/" not in line and "/usr/local/" not in line, line
        assert "/torch-macos-build/" not in line, line
print(torch.__config__.show())
PY
.venv/bin/python "$source_root/test/test_security_backports.py" -v 2>&1 | tee "$source_root/dist/native-regressions.log"
cd "$source_root"
printf '%s\n' 'Native CPU regressions passed. MPS is built; GPU kernels and generation were not executed.' >> dist/build-provenance.txt
