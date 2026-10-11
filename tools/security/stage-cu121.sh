#!/usr/bin/env bash
set -euo pipefail

checkpoint_dir="$RUNNER_TEMP/torch-checkpoint"
image_tag="torch-security-stage:$GITHUB_SHA"
container_name="torch-security-build-$BUILD_STAGE"
base_image='nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:327c9e046fbf662275be0934742f7e5412f9b24402ee90bf4d649c1a21707912'

if [[ -n "$CHECKPOINT_ARTIFACT" ]]; then
  (
    cd "$checkpoint_dir"
    sha256sum --check SHA256SUMS
    test "$(cat source-revision.txt)" = "$GITHUB_SHA"
  )
  available=$(df --output=avail -B1 "$PWD" | tail -1)
  required=$(cat "$checkpoint_dir/uncompressed-bytes")
  test "$available" -gt "$((required + 5 * 1024 * 1024 * 1024))"
  zstd --decompress --stdout "$checkpoint_dir/image.tar.zst" | docker load
  zstd --decompress --stdout "$checkpoint_dir/source.tar.zst" | sudo tar --extract --preserve-permissions -C "$PWD"
  rm "$checkpoint_dir/image.tar.zst" "$checkpoint_dir/source.tar.zst"
else
  docker pull "$base_image"
  docker tag "$base_image" "$image_tag"
fi

set +e
docker run --name "$container_name" \
  --volume "$PWD:/source" \
  --volume "$(command -v uv):/usr/local/bin/uv:ro" \
  --workdir /source \
  --env TORCH_SECURITY_BUILD_SECONDS=10800 \
  --env EXPECTED_SOURCE_REVISION="$GITHUB_SHA" \
  "$image_tag" bash -euo pipefail -c '
    if [[ -f build/build.ninja ]]; then
      export PATH=/tmp/torch-security-build/.venv/bin:$PATH
      wc -l build/.ninja_log
      ninja -C build -n -d explain > /source/security-resume-plan.log 2>&1
      tail -60 /source/security-resume-plan.log
    fi
    exec bash tools/security/build-cu121.sh
  '
build_status=$?
set -e

if [[ "$build_status" == 0 ]]; then
  printf 'wheel=torch-cu121-security-%s-stage-%s\n' "$GITHUB_SHA" "$BUILD_STAGE" >> "$GITHUB_OUTPUT"
  exit 0
fi
if [[ "$build_status" != 75 ]]; then
  exit "$build_status"
fi
if [[ "$BUILD_STAGE" -ge 4 ]]; then
  printf '%s\n' 'The CUDA wheel did not complete within four build stages.' >&2
  exit 1
fi

# A stopped container preserves the compiler, Python, headers, and environment paths.
test "$(docker inspect --format '{{.State.Running}}' "$container_name")" = false
docker commit "$container_name" "$image_tag"
docker image inspect --format '{{.Size}}' "$image_tag" > "$checkpoint_dir/image-bytes"
source_bytes=$(sudo du -sb "$PWD" | cut -f1)
image_bytes=$(cat "$checkpoint_dir/image-bytes")
printf '%s\n' "$((source_bytes + image_bytes))" > "$checkpoint_dir/uncompressed-bytes"
printf '%s\n' "$GITHUB_SHA" > "$checkpoint_dir/source-revision.txt"
docker save "$image_tag" | zstd --threads=2 -3 -o "$checkpoint_dir/image.tar.zst"
sudo tar --create --format=pax --pax-option=delete=atime,delete=ctime \
  --exclude=.git -C "$PWD" . | zstd --threads=2 -3 -o "$checkpoint_dir/source.tar.zst"
(
  cd "$checkpoint_dir"
  sha256sum image.tar.zst source.tar.zst source-revision.txt uncompressed-bytes > SHA256SUMS
)
du -h "$checkpoint_dir/"*
printf 'checkpoint=torch-cu121-checkpoint-%s-stage-%s\n' "$GITHUB_SHA" "$BUILD_STAGE" >> "$GITHUB_OUTPUT"
