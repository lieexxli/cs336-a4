wget --timeout=10 \
  --tries=2 \
  --wait=1 \
  --random-wait \
  --user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36" \
  -i data/wiki/subsampled_positive_urls.txt \
  --warc-file=data/wiki/unfiltered_positive_samples \
  -O /dev/null
