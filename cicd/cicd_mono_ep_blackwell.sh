#!/bin/bash
# mono-ep package kernel tests on one B200 (packages/mono-ep/tests marked gpu_blackwell). Kept apart from
# cicd/cicd_blackwell.sh so the Axolotl suite and the package suite run in separate containers. Not
# testmon-selected and no coverage upload (coverage tracks src/axolotl).
set -euo pipefail

python - <<'BLACKWELL_ENV'
import os

import torch

assert os.environ["PYTORCH_VERSION"] in torch.__version__, torch.__version__
major, minor = torch.cuda.get_device_capability()
assert major >= 10, f"mono-ep Blackwell tests need an SM100+ GPU, got sm{major}{minor}"
print(f"{torch.cuda.get_device_name()} sm{major}{minor}", flush=True)
BLACKWELL_ENV

bash cicd/install_mono_ep.sh
uv pip install --python "$(command -v python)" "./packages/mono-ep[sonic,sonic-fp8-experimental]"

rm -f mono-ep-blackwell-junit.xml
# The package's gpu_blackwell tests minus the slow ones; the gpu-only package tests run on the L40S
# kernels lane (cicd/cicd_cuda_kernels.sh). -v so cicd/verify_changed_tests.py sees each outcome.
python -m pytest -v --durations=10 packages/mono-ep/tests -m "gpu_blackwell and not slow" \
  --junitxml=mono-ep-blackwell-junit.xml
