#!/usr/bin/env bash
# Build the release AppImage in an Ubuntu 22.04 container.
#
# The host does not need GTK/WebKit development packages, Rust, or Node: the
# Docker build installs the same toolchain and dependencies as the CI workflow.
# The finished AppImage is copied to release/ and checked against glibc 2.35.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${APPIMAGE_DOCKER_IMAGE:-mikrotitan-appimage-builder:local}"
DOCKERFILE="${APPIMAGE_DOCKERFILE:-$ROOT/Dockerfile.appimage}"
OUT="${1:-$ROOT/release}"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker is required to use this target" >&2
  exit 1
fi

mkdir -p "$OUT"
TMP="$(mktemp -d)"
CID=""
cleanup() {
  if [[ -n "$CID" ]]; then
    docker rm -f "$CID" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP"
}
trap cleanup EXIT

echo "Building the release AppImage in Ubuntu 22.04 (glibc 2.35)"
echo "  image:      $IMAGE"
echo "  Dockerfile: $DOCKERFILE"
echo "  output:     $OUT"
echo

docker build --file "$DOCKERFILE" --tag "$IMAGE" "$ROOT"

echo
echo "Extracting the AppImage from the build image..."
# Use docker cp rather than a bind mount: Docker Desktop only shares selected
# host paths, and the temporary extraction directory may not be one of them.
CID="$(docker create "$IMAGE")"
docker cp "$CID:/artifacts/." "$TMP"

mapfile -t built < <(find "$TMP" -maxdepth 1 -type f -name '*.AppImage' -print | sort)
if ((${#built[@]} == 0)); then
  echo "error: the container did not produce an AppImage" >&2
  exit 1
fi

for artifact in "${built[@]}"; do
  name="$(basename "$artifact")"
  cp -f "$artifact" "$OUT/$name"
  chmod +x "$OUT/$name"
  bash "$ROOT/scripts/check-appimage-glibc.sh" "$OUT/$name"
done

echo
echo "Local AppImage(s) written to $OUT:"
for artifact in "${built[@]}"; do
  printf '  %s\n' "$OUT/$(basename "$artifact")"
done
