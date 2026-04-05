SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DELETE_ON_ERROR:

UV_RUN ?= uv run
WGET ?= wget

ROUTE ?= default
ROUTE_ROOT ?= artifacts/$(ROUTE)
ROUTE_DATA_DIR ?= $(ROUTE_ROOT)/data
ROUTE_STAMP_DIR ?= $(ROUTE_ROOT)/.stamps
ROUTE_COMMONCRAWL_DIR ?= $(ROUTE_DATA_DIR)/commoncrawl
ROUTE_CC_DIR ?= $(ROUTE_DATA_DIR)/CC
ROUTE_RAW_DIR ?= $(ROUTE_DATA_DIR)/raw
ROUTE_WIKI_DIR ?= $(ROUTE_DATA_DIR)/wiki
ROUTE_LEADERBOARD_DIR ?= $(ROUTE_DATA_DIR)/leaderboard
ROUTE_LEADERBOARD_CLASSIFIER_DIR ?= $(ROUTE_LEADERBOARD_DIR)/classifier
ROUTE_OUT_MODELS_DIR ?= $(ROUTE_ROOT)/out/models
ROUTE_TOKENS_BIN ?= $(ROUTE_DATA_DIR)/tokens.bin

SHARED_DATA_DIR := data
SHARED_STAMP_DIR := $(SHARED_DATA_DIR)/.shared-stamps
CLASSIFIERS_DIR := $(SHARED_DATA_DIR)/classifiers
SHARED_WIKI_DIR := $(SHARED_DATA_DIR)/wiki
COMMONCRAWL_DIR := $(SHARED_DATA_DIR)/commoncrawl
CC_CACHE_DIR := $(COMMONCRAWL_DIR)/cache/CC
WET_CACHE_DIR := $(COMMONCRAWL_DIR)/cache/raw
PALOMA_DIR := $(SHARED_DATA_DIR)/paloma

NSFW_MODEL := $(CLASSIFIERS_DIR)/dolma_fasttext_nsfw_jigsaw_model.bin
TOXIC_MODEL := $(CLASSIFIERS_DIR)/dolma_fasttext_hatespeech_jigsaw_model.bin
LID_MODEL := $(CLASSIFIERS_DIR)/lid.176.bin
WIKI_TITLES := $(SHARED_WIKI_DIR)/titles.gz
WIKI_URLS := $(SHARED_WIKI_DIR)/enwiki-20240420-extracted_urls.txt.gz
PALOMA_BIN := $(PALOMA_DIR)/tokenized_paloma_c4_100_domains_validation.bin
ROUTE_PALOMA_LINK := $(ROUTE_LEADERBOARD_DIR)/tokenized_paloma_c4_100_domains_validation.bin
WARC_PATHS := $(COMMONCRAWL_DIR)/warc.paths.gz
WET_PATHS := $(COMMONCRAWL_DIR)/wet.paths.gz
ROOT_MAKEFILE := $(firstword $(MAKEFILE_LIST))

define run_make_timed
start=$$(date +%s); \
printf "==> %s\n" "$(1)"; \
if $(MAKE) --no-print-directory -f "$(ROOT_MAKEFILE)" $(2); then \
  end=$$(date +%s); \
  elapsed=$$((end - start)); \
  printf "<== %s finished in %dm %02ds\n" "$(1)" $$((elapsed / 60)) $$((elapsed % 60)); \
else \
  status=$$?; \
  end=$$(date +%s); \
  elapsed=$$((end - start)); \
  printf "<!! %s failed after %dm %02ds\n" "$(1)" $$((elapsed / 60)) $$((elapsed % 60)); \
  exit $$status; \
fi
endef

.PHONY: setup classifiers wiki-urls paloma route-layout

setup: $(SHARED_STAMP_DIR)/setup.done

classifiers: $(NSFW_MODEL) $(TOXIC_MODEL) $(LID_MODEL)

wiki-urls: $(WIKI_TITLES) $(WIKI_URLS)

paloma: $(PALOMA_BIN)

route-layout: $(ROUTE_PALOMA_LINK)

$(SHARED_STAMP_DIR)/dirs.done:
	mkdir -p \
	  $(SHARED_STAMP_DIR) \
	  $(CLASSIFIERS_DIR) \
	  $(SHARED_WIKI_DIR) \
	  $(COMMONCRAWL_DIR) \
	  $(CC_CACHE_DIR) \
	  $(WET_CACHE_DIR) \
	  $(PALOMA_DIR) \
	  $(SHARED_DATA_DIR)
	touch $@

$(ROUTE_STAMP_DIR)/dirs.done:
	mkdir -p \
	  $(ROUTE_STAMP_DIR) \
	  $(ROUTE_COMMONCRAWL_DIR) \
	  $(ROUTE_CC_DIR) \
	  $(ROUTE_RAW_DIR) \
	  $(ROUTE_WIKI_DIR) \
	  $(ROUTE_LEADERBOARD_DIR) \
	  $(ROUTE_LEADERBOARD_CLASSIFIER_DIR) \
	  $(ROUTE_OUT_MODELS_DIR)
	touch $@

$(SHARED_STAMP_DIR)/setup.done: pyproject.toml uv.lock | $(SHARED_STAMP_DIR)/dirs.done
	uv sync
	$(UV_RUN) python -c 'import nltk; nltk.download("punkt_tab")'
	touch $@

$(NSFW_MODEL): | $(SHARED_STAMP_DIR)/dirs.done
	$(WGET) -O $@ \
	  "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin"

$(TOXIC_MODEL): | $(SHARED_STAMP_DIR)/dirs.done
	$(WGET) -O $@ \
	  "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin"

$(LID_MODEL): | $(SHARED_STAMP_DIR)/dirs.done
	$(WGET) -O $@ \
	  "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

$(WIKI_TITLES): | $(SHARED_STAMP_DIR)/dirs.done
	$(WGET) -O $@ \
	  "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz"

$(WIKI_URLS): $(WIKI_TITLES) | $(SHARED_STAMP_DIR)/dirs.done
	zcat $< \
	  | tail -n +2 \
	  | awk '{print "https://en.wikipedia.org/wiki/"$$1}' \
	  | gzip > $@

$(PALOMA_BIN): make/build_paloma_bin.py | $(SHARED_STAMP_DIR)/setup.done $(SHARED_STAMP_DIR)/dirs.done
	$(UV_RUN) python make/build_paloma_bin.py

$(ROUTE_PALOMA_LINK): $(PALOMA_BIN) | $(ROUTE_STAMP_DIR)/dirs.done
	ln -sf $(abspath $(PALOMA_BIN)) $@

$(WARC_PATHS): make/fetch_cached.sh | $(SHARED_STAMP_DIR)/dirs.done
	bash make/fetch_cached.sh \
	  "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/warc.paths.gz" \
	  $@ \
	  gzip

$(WET_PATHS): make/fetch_cached.sh | $(SHARED_STAMP_DIR)/dirs.done
	bash make/fetch_cached.sh \
	  "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/wet.paths.gz" \
	  $@ \
	  gzip
