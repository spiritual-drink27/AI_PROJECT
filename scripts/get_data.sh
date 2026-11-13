#!/usr/bin/env bash
# Download the BASIL dataset (Fan et al., 2019) into data/raw/BASIL.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
if [ -d data/raw/BASIL/articles ]; then
  echo "BASIL already present in data/raw/BASIL"
else
  git clone --depth 1 https://github.com/launchnlp/BASIL data/raw/BASIL
fi
