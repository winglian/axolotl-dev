#!/bin/bash
set -euo pipefail

uv pip install --python "$(command -v python)" setuptools wheel packaging ninja
apt-get update
apt-get install -y libdw-dev libelf-dev software-properties-common
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install -y g++-13
python packages/mono-ep/tools/fetch_build_dependencies.py
DG_USE_LOCAL_VERSION=0 DG_FORCE_BUILD=1 CC=gcc-13 CXX=g++-13 MAX_JOBS=4 \
  uv pip install --python "$(command -v python)" --no-build-isolation --no-deps packages/mono-ep/vendor/deepgemm
uv pip install --python "$(command -v python)" --no-deps packages/mono-ep
