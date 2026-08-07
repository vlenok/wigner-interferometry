#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repository_dir="$(dirname -- "$script_dir")"
image_name="${WIGNER_AUTORESEARCH_IMAGE:-wigner-autoresearch:local}"
results_file="${WIGNER_AUTORESEARCH_RESULTS:-$script_dir/results.jsonl}"

if ! docker image inspect "$image_name" >/dev/null 2>&1; then
    echo "Missing $image_name. Build it with:" >&2
    echo "  docker build -f autoresearch/Dockerfile -t $image_name ." >&2
    exit 1
fi

git_commit="$(git -C "$repository_dir" rev-parse --short HEAD)"
mkdir -p "$(dirname -- "$results_file")"

docker run --rm \
    --network none \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --pids-limit 64 \
    --cpus 1.0 \
    --memory 4g \
    --memory-swap 4g \
    --tmpfs /tmp:rw,nosuid,nodev,noexec,size=2g,mode=1777 \
    --user 65532:65532 \
    --env "AUTORESEARCH_GIT_COMMIT=$git_commit" \
    --volume "$script_dir/candidate.py:/workspace/autoresearch/candidate.py:ro" \
    "$image_name" "$@" | tee -a "$results_file"
