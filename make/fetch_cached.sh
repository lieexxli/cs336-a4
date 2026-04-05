#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: fetch_cached.sh <url> <dest> [size|gzip]" >&2
  exit 1
fi

url=$1
dest=$2
validate_mode=${3:-size}
tmp="${dest}.part"
ok="${dest}.ok"

validate_file() {
  local path=$1
  case "$validate_mode" in
    gzip)
      gzip -t "$path"
      ;;
    size)
      [[ -s "$path" ]]
      ;;
    *)
      echo "unknown validation mode: $validate_mode" >&2
      exit 1
      ;;
  esac
}

mkdir -p "$(dirname "$dest")"

if [[ -f "$dest" ]]; then
  if [[ -f "$ok" ]]; then
    printf "Reusing verified cache %s\n" "$dest"
    exit 0
  fi

  if validate_file "$dest"; then
    touch "$ok"
    printf "Validated existing cache %s\n" "$dest"
    exit 0
  fi

  printf "Removing invalid cache %s\n" "$dest"
  rm -f "$dest"
fi

if [[ -f "$tmp" ]]; then
  printf "Resuming partial download %s\n" "$tmp"
else
  printf "Downloading %s\n" "$dest"
fi

wget --continue --output-document="$tmp" "$url"
validate_file "$tmp"
mv -f "$tmp" "$dest"
touch "$ok"
printf "Cached %s\n" "$dest"
