#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH="${1:-/Volumes/Crucial/AI/DATASETS/GenImage_faces_keep_struct}"

if [[ ! -d "$ROOT_PATH" ]]; then
  echo "Directory not found: $ROOT_PATH" >&2
  exit 1
fi

rename_dirs() {
  local from="$1"
  local to="$2"

  while IFS= read -r -d '' dir; do
    local parent
    parent="$(dirname "$dir")"
    local target="$parent/$to"

    if [[ -e "$target" ]]; then
      echo "Skipping rename of '$dir' because '$target' already exists"
      continue
    fi

    mv -- "$dir" "$target"
    echo "Renamed '$dir' → '$target'"
  done < <(find "$ROOT_PATH" -depth -type d -name "$from" -print0)
}

rename_dirs ai fake
rename_dirs nature real
