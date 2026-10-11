#!/usr/bin/env bash
set -euo pipefail

[[ "$CHECKPOINT_RUN_ID" =~ ^[0-9]+$ ]]
[[ "$CHECKPOINT_SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]
[[ "$CHECKPOINT_ARTIFACT" == "torch-cu121-checkpoint-$CHECKPOINT_SOURCE_REVISION-stage-"[1-3] ]]
git merge-base --is-ancestor "$CHECKPOINT_SOURCE_REVISION" "$GITHUB_SHA"
checkpoint_dir="$RUNNER_TEMP/torch-checkpoint"
gh api "repos/$GITHUB_REPOSITORY/actions/runs/$CHECKPOINT_RUN_ID" > "$RUNNER_TEMP/checkpoint-run.json"
jq --exit-status --arg sha "$CHECKPOINT_SOURCE_REVISION" --arg repo "$GITHUB_REPOSITORY" '
  .head_sha == $sha and .repository.full_name == $repo and
  .path == ".github/workflows/security-cu121-resumable.yml" and
  .head_branch == "security/cu121-backports" and (.event == "push" or .event == "workflow_dispatch")
' "$RUNNER_TEMP/checkpoint-run.json"
stage_number="${CHECKPOINT_ARTIFACT##*-}"
gh api "repos/$GITHUB_REPOSITORY/actions/runs/$CHECKPOINT_RUN_ID/jobs?per_page=100" |
  jq --exit-status --arg name "stage-$stage_number / stage" '
    [.jobs[] | select(.name == $name and .conclusion == "success")] | length == 1
  '
gh api "repos/$GITHUB_REPOSITORY/actions/runs/$CHECKPOINT_RUN_ID/artifacts?per_page=100" |
  jq --exit-status --arg name "$CHECKPOINT_ARTIFACT" '
    [.artifacts[] | select(.name == $name and .expired == false)] |
    if length == 1 then .[0] else error("Expected one unexpired checkpoint") end
  ' > "$RUNNER_TEMP/checkpoint-artifact.json"
artifact_id=$(jq --raw-output --exit-status '.id' "$RUNNER_TEMP/checkpoint-artifact.json")
digest=$(jq --raw-output --exit-status '.digest' "$RUNNER_TEMP/checkpoint-artifact.json")
[[ "$artifact_id" =~ ^[0-9]+$ ]]
[[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]
archive="$RUNNER_TEMP/torch-checkpoint.zip"
gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$artifact_id/zip" > "$archive"
printf '%s  %s\n' "${digest#sha256:}" "$archive" | sha256sum --check
python3 - "$archive" "$checkpoint_dir" <<'PY'
import stat
import sys
import zipfile

expected = {"image-bytes", "uncompressed-bytes", "source-revision.txt", "image.tar.zst", "source.tar.zst", "SHA256SUMS"}
with zipfile.ZipFile(sys.argv[1]) as archive:
    entries = archive.infolist()
    if len(entries) != len(expected) or {entry.filename for entry in entries} != expected:
        raise ValueError("Unexpected checkpoint archive entries")
    if any(stat.S_ISLNK(entry.external_attr >> 16) for entry in entries):
        raise ValueError("Checkpoint archive contains a symbolic link")
    archive.extractall(sys.argv[2])
PY
rm "$archive"
