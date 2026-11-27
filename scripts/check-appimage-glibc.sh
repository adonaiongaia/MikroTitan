#!/usr/bin/env bash
# Fail if the AppImage payload needs a newer glibc than our supported baseline.
#
# An AppImage does not bundle glibc, so the minimum glibc is fixed by the
# machine that built it. The AppImageHub catalogue tests on the oldest Ubuntu
# LTS still in standard support, which is why release builds use Ubuntu 22.04
# and its glibc 2.35 baseline.
#
# Usage: scripts/check-appimage-glibc.sh path/to/AppImage
#
# APPIMAGE_MAX_GLIBC can override the baseline for a local-only build on a
# newer host. Never use that override for a release artifact.
set -euo pipefail

MAX_GLIBC="${APPIMAGE_MAX_GLIBC:-2.35}"

if [[ $# -ne 1 || ! -f "$1" ]]; then
  echo "usage: $0 path/to/AppImage" >&2
  exit 2
fi
APPIMAGE="$(realpath "$1")"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Checking $(basename "$APPIMAGE") against glibc $MAX_GLIBC"
(
  cd "$WORK"
  "$APPIMAGE" --appimage-extract >/dev/null
)

APPDIR="$WORK/squashfs-root"
if [[ ! -d "$APPDIR" ]]; then
  echo "error: AppImage did not extract a squashfs-root directory" >&2
  exit 2
fi

version_gt() {
  [[ "$1" != "$2" ]] &&
    [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n 1)" == "$1" ]]
}

highest=""
failures=()

while IFS= read -r -d '' file; do
  # Only ELF files carry versioned glibc symbol references.
  [[ "$(head -c 4 "$file" | od -An -tx1 | tr -d ' \n')" == "7f454c46" ]] || continue

  while IFS= read -r version; do
    [[ -n "$version" ]] || continue
    if [[ -z "$highest" ]] || version_gt "$version" "$highest"; then
      highest="$version"
    fi
    if version_gt "$version" "$MAX_GLIBC"; then
      failures+=("${file#"$APPDIR"/}: GLIBC_$version")
    fi
  done < <(
    objdump -T "$file" 2>/dev/null |
      grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' |
      sed 's/^GLIBC_//' |
      sort -u
  )
done < <(find "$APPDIR" -type f \( -name '*.so' -o -name '*.so.*' -o -perm -u+x \) -print0)

if ((${#failures[@]})); then
  printf 'FAIL %s\n' "${failures[@]}" >&2
  cat >&2 <<EOF
error: the AppImage requires a glibc newer than $MAX_GLIBC.
glibc is only as old as the build host: rebuild on Ubuntu 22.04 with the
"Build AppImage" workflow instead of a newer distribution.
highest version found: ${highest:-unknown}
EOF
  exit 1
fi

echo "OK: highest bundled glibc reference is ${highest:-none} (baseline $MAX_GLIBC)"
