#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${ASSETS_DIR:-$SCRIPT_DIR/cs336_data/assets}"
mkdir -p "$ASSETS_DIR"

download() {
    local name="$1" url="$2" target="$ASSETS_DIR/$1"
    if [[ -s "$target" ]]; then
        printf 'Already present: %s\n' "$target"
        return
    fi
    curl --fail --location --retry 3 "$url" --output "$target.part"
    mv -- "$target.part" "$target"
    printf 'Downloaded: %s\n' "$target"
}

command -v curl >/dev/null || { echo 'curl is required' >&2; exit 1; }
download dolma_fasttext_nsfw_jigsaw_model.bin \
  https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin
download dolma_fasttext_hatespeech_jigsaw_model.bin \
  https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin
