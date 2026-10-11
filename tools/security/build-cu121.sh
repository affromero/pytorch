#!/usr/bin/env bash
set -euo pipefail

export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/source

export DEBIAN_FRONTEND=noninteractive
if [[ ! -f /tmp/torch-security-toolchain-ready ]]; then
  apt-get update
  apt-get install -y --no-install-recommends build-essential ca-certificates git libopenblas-dev patchelf pkg-config
  rm -rf /var/lib/apt/lists/*
  touch /tmp/torch-security-toolchain-ready
fi
if [[ -n "${EXPECTED_SOURCE_REVISION:-}" ]]; then
  test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE_REVISION"
fi

build_env=/tmp/torch-security-build
mkdir -p "$build_env"
cat > "$build_env/pyproject.toml" <<'TOML'
[project]
name = "torch-security-build"
version = "0.0.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "setuptools==69.5.1", "wheel==0.43.0", "cmake==3.29.6", "ninja==1.11.1.1",
  "numpy==1.26.4", "pyyaml==6.0.2", "packaging==24.2", "typing-extensions>=4.8",
  "auditwheel==6.2.0",
  "nvidia-cuda-nvrtc-cu12==12.1.105", "nvidia-cuda-runtime-cu12==12.1.105",
  "nvidia-cuda-cupti-cu12==12.1.105", "nvidia-cudnn-cu12==8.9.2.26",
  "nvidia-cublas-cu12==12.1.3.1", "nvidia-cufft-cu12==11.0.2.54",
  "nvidia-curand-cu12==10.3.2.106", "nvidia-cusolver-cu12==11.4.5.107",
  "nvidia-cusparse-cu12==12.1.0.106", "nvidia-nccl-cu12==2.20.5",
  "nvidia-nvtx-cu12==12.1.105", "triton==2.3.0",
]
TOML
if [[ -f "$build_env/uv.lock" ]]; then
  uv sync --project "$build_env" --python 3.11.15 --frozen
else
  uv sync --project "$build_env" --python 3.11.15
fi
export PATH="$build_env/.venv/bin:$PATH"
export VIRTUAL_ENV="$build_env/.venv"
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=2
export USE_CUDA=1 USE_CUDNN=1 USE_DISTRIBUTED=1 USE_SYSTEM_NCCL=1
export BUILD_TEST=0 BUILD_CAFFE2=0 CMAKE_BUILD_TYPE=Release
export TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.9;9.0'
export _GLIBCXX_USE_CXX11_ABI=0
export PYTORCH_BUILD_VERSION=2.3.0+cu121.security1
export PYTORCH_BUILD_NUMBER=1
nvidia_dir="$build_env/.venv/lib/python3.11/site-packages/nvidia"
export CUDNN_INCLUDE_DIR="$nvidia_dir/cudnn/include"
export CUDNN_LIB_DIR="$nvidia_dir/cudnn/lib"
export NCCL_INCLUDE_DIR="$nvidia_dir/nccl/include"
export NCCL_LIB_DIR="$nvidia_dir/nccl/lib"
test -L "$CUDNN_LIB_DIR/libcudnn.so" || ln -s libcudnn.so.8 "$CUDNN_LIB_DIR/libcudnn.so"
test -L "$NCCL_LIB_DIR/libnccl.so" || ln -s libnccl.so.2 "$NCCL_LIB_DIR/libnccl.so"
export PYTORCH_EXTRA_INSTALL_REQUIREMENTS='nvidia-cuda-nvrtc-cu12==12.1.105|nvidia-cuda-runtime-cu12==12.1.105|nvidia-cuda-cupti-cu12==12.1.105|nvidia-cudnn-cu12==8.9.2.26|nvidia-cublas-cu12==12.1.3.1|nvidia-cufft-cu12==11.0.2.54|nvidia-curand-cu12==10.3.2.106|nvidia-cusolver-cu12==11.4.5.107|nvidia-cusparse-cu12==12.1.0.106|nvidia-nccl-cu12==2.20.5|nvidia-nvtx-cu12==12.1.105|triton==2.3.0'
if [[ -n "${TORCH_SECURITY_BUILD_SECONDS:-}" ]]; then
  set +e
  timeout --signal=INT --kill-after=60s "$TORCH_SECURITY_BUILD_SECONDS" python setup.py bdist_wheel
  build_status=$?
  set -e
  if [[ "$build_status" == 124 ]]; then
    exit 75
  fi
  if [[ "$build_status" != 0 ]]; then
    exit "$build_status"
  fi
else
  python setup.py bdist_wheel
fi

auditwheel repair dist/torch-*.whl --plat manylinux_2_35_x86_64 \
  --exclude 'libcu*.so*' --exclude 'libnccl.so*' --exclude 'libnv*.so*' \
  --wheel-dir repaired
rm dist/torch-*.whl
mv repaired/*.whl dist/

test_env=/tmp/torch-security-wheel-test
mkdir -p "$test_env"
wheel_path=$(realpath dist/torch-*.whl)
cat > "$test_env/pyproject.toml" <<TOML
[project]
name = "torch-security-wheel-test"
version = "0.0.0"
requires-python = ">=3.11,<3.12"
dependencies = ["torch", "numpy==1.26.4"]
[tool.uv.sources]
torch = { path = "$wheel_path" }
TOML
uv sync --project "$test_env" --python 3.11.15
source_root=$PWD
cd "$test_env"
"$test_env/.venv/bin/python" - <<'PY'
import torch
assert torch.__version__ == "2.3.0+cu121.security1", torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert not torch._C._GLIBCXX_USE_CXX11_ABI
assert torch.backends.cudnn.version() == 8902, torch.backends.cudnn.version()
assert torch.backends.mkldnn.is_available()
assert torch.cuda.device_count() == 0
print(torch.__config__.show())
PY
"$test_env/.venv/bin/python" "$source_root/test/test_security_backports.py" -v 2>&1 | tee "$source_root/dist/native-regressions.log"
cd "$source_root"
{
  git rev-parse HEAD
  nvcc --version
  python --version
  printf '%s\n' 'CPU regressions passed. GPU kernel execution and generation were not tested.'
} > dist/build-provenance.txt
sha256sum dist/*.whl > dist/SHA256SUMS
