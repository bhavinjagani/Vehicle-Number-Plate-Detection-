#!/usr/bin/env bash
# Clone YOLOv5 and YOLOv9 repositories at known-good commits.
# Optionally installs their native requirements into the current environment.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT}/external"
PYTHON_BIN="${PYTHON:-python3}"
SKIP_PIP=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-pip)
      SKIP_PIP=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

mkdir -p "${EXTERNAL_DIR}"

clone_or_update() {
  local repo_name=$1
  local repo_url=$2
  local repo_ref=$3
  local dest="${EXTERNAL_DIR}/${repo_name}"

  if [[ -d "${dest}/.git" ]]; then
    echo "[+] Updating ${repo_name}..."
    git -C "${dest}" fetch --all --tags --prune
    git -C "${dest}" checkout "${repo_ref}"
    git -C "${dest}" pull --ff-only || true
  else
    echo "[+] Cloning ${repo_name}..."
    git clone --depth=1 --branch "${repo_ref}" "${repo_url}" "${dest}"
  fi
}

install_requirements() {
  local dest=$1
  local req_file="${dest}/requirements.txt"
  if [[ "${SKIP_PIP}" == true ]]; then
    return
  fi
  if [[ -f "${req_file}" ]]; then
    echo "[+] Installing requirements for ${dest}..."
    "${PYTHON_BIN}" -m pip install -r "${req_file}"
  else
    echo "[!] No requirements.txt found for ${dest}, skipping."
  fi
}

clone_or_update "yolov5" "https://github.com/ultralytics/yolov5.git" "v7.0"
clone_or_update "yolov9" "https://github.com/WongKinYiu/yolov9.git" "main"

install_requirements "${EXTERNAL_DIR}/yolov5"
install_requirements "${EXTERNAL_DIR}/yolov9"

cat <<EOF
Done.
Repos installed under: ${EXTERNAL_DIR}

Next:
  1. Update configs/datasets.yaml with your local dataset paths.
  2. python src/data/prepare_dataset.py --config configs/datasets.yaml
  3. Start training with: python src/train.py --model {yolov5|yolov9} ...
EOF
