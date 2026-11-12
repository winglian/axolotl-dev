#!/bin/bash
set -e

export COVERAGE_CORE=sysmon

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, f'Expected torch $PYTORCH_VERSION but got {torch.__version__}'"

bash ./cicd/prepare_hf_cache.sh

uv pip install --python "$(command -v python)" --no-deps packages/mono-ep
python -m pytest -q packages/mono-ep/tests -m gpu
python -m pytest -q packages/mono-ep/tests/test_api.py packages/mono-ep/tests/test_quantization.py packages/mono-ep/tests/test_mono_ep.py packages/mono-ep/tests/test_persistent_selection.py packages/mono-ep/tests/test_prepare.py packages/mono-ep/tests/test_sonic_loader.py -m "not gpu"

# Keep process-wide model patching isolated from kernel parity tests.
python cicd/pytest_testmon.py -v --durations=10 --maxfail=10 \
  /workspace/axolotl/tests/e2e/kernels/lora_patching/ \
  --cov=axolotl

# gpu_blackwell tests run on B200 in cicd/cicd_blackwell.sh.
python cicd/feature_testmon.py kernels -v --durations=10 --maxfail=10 -m "not slow and not gpu_blackwell" \
  --cov=axolotl \
  --cov-append

python cicd/pytest_testmon.py -v --durations=10 --maxfail=10 -m slow \
  /workspace/axolotl/tests/monkeypatch/test_fla_mamba.py \
  --cov=axolotl \
  --cov-append \
  --cov-report=xml:e2e-kernel-coverage.xml

if [ "${TESTMON_FULL:-}" = "true" ]; then
  codecovcli upload-process -t "$CODECOV_TOKEN" -f e2e-kernel-coverage.xml -F e2e,kernels,pytorch-${PYTORCH_VERSION} || true
fi
