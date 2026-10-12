#!/bin/bash
# Download and verify the OpenNeuro-derived test bundle from its GitHub release.
#   scripts/fetch_testdata.sh [dest]      (default: $UNPILE_TESTDATA or ./testdata/real)
set -euo pipefail
REPO=${UNPILE_TESTDATA_REPO:-PennLINC/unpile-testdata}
TAG=${UNPILE_TESTDATA_TAG:-v1}
SHA256=${UNPILE_TESTDATA_SHA256:-0aaf2feb55503a08362b034126449eb8b6fa0536f65b8cc6bad410f5f0e5f601}
DEST=${1:-${UNPILE_TESTDATA:-testdata/real}}
mkdir -p "$DEST"
tarball="$DEST/unpile-testdata-$TAG.tar.gz"
if [ ! -d "$DEST/$TAG" ]; then
  curl -fsSL -o "$tarball" "https://github.com/$REPO/releases/download/$TAG/unpile-testdata-$TAG.tar.gz"
  echo "$SHA256  $tarball" | sha256sum -c -
  tar xzf "$tarball" -C "$DEST"
  rm -f "$tarball"
fi
# registry.txt is "<path> <sha256>" per line (TRXScan/pooch order), so verify by hand
cd "$DEST/$TAG"
bad=0
while read -r path hash; do
  [ -z "$path" ] && continue
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  [ "$actual" = "$hash" ] || { echo "checksum mismatch: $path" >&2; bad=1; }
done < registry.txt
[ $bad -eq 0 ] && echo "testdata $TAG verified in $DEST/$TAG"
