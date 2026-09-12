#!/usr/bin/env bash
# Build a clean downloadable zip of the copper ensemble strategy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-0.2.0}"
NAME="copper-ensemble-allgreen-dd30-v${VERSION}"
OUT_DIR="${ROOT}/releases"
STAGE="$(mktemp -d)"
DEST="${STAGE}/${NAME}"

mkdir -p "${DEST}" "${OUT_DIR}"

# Portable copy without rsync
shopt -s dotglob nullglob
for item in "${ROOT}"/*; do
  base="$(basename "${item}")"
  case "${base}" in
    releases|dist|build|.venv|venv|.git|__pycache__|.pytest_cache|*.egg-info|PUBLIC_URL|*.log)
      continue
      ;;
  esac
  cp -a "${item}" "${DEST}/"
done
# drop caches inside the staged tree
find "${DEST}" -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '*.egg-info' \) -prune -exec rm -rf {} +
find "${DEST}" -type f \( -name '*.pyc' -o -name '*.log' \) -delete

test -f "${DEST}/README.md"
test -f "${DEST}/DOWNLOAD.md"
test -f "${DEST}/config/default.yaml"
test -f "${DEST}/setup.py"

ZIP_PATH="${OUT_DIR}/${NAME}.zip"
rm -f "${ZIP_PATH}"
(
  cd "${STAGE}"
  zip -qr "${ZIP_PATH}" "${NAME}"
)

(
  cd "${OUT_DIR}"
  sha256sum "${NAME}.zip" > "${NAME}.zip.sha256"
)

if [[ -d /opt/cursor/artifacts ]]; then
  cp -f "${ZIP_PATH}" "/opt/cursor/artifacts/${NAME}.zip"
  cp -f "${OUT_DIR}/${NAME}.zip.sha256" "/opt/cursor/artifacts/${NAME}.zip.sha256"
fi

rm -rf "${STAGE}"
echo "Built ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
cat "${OUT_DIR}/${NAME}.zip.sha256"
