#!/usr/bin/env bash
set -euo pipefail

URL_LIST=${1:-data/wiki/subsampled_positive_urls.txt}
WARC_PREFIX=${2:-data/wiki/unfiltered_positive_samples}

wget --timeout=10 \
  --tries=2 \
  --wait=1 \
  --random-wait \
  --user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36" \
  -i "$URL_LIST" \
  --warc-file="$WARC_PREFIX" \
  -O /dev/null
