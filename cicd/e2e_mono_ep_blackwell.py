"""Modal app to run the mono-ep package's gpu_blackwell tests on one B200 (set GPU_TYPE=B200)."""

from .modal_testmon import run_local, run_remote
from .single_gpu import GPU_CONFIG, VOLUME_CONFIG, app, cicd_image, refresh_source


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=60 * 60,
    cpu=8.0,
    memory=131072,
    volumes=VOLUME_CONFIG,
)
def cicd_mono_ep_blackwell(state: dict[str, bytes], full: bool, select: bool):
    # Not testmon-driven: run_remote's TESTMON_* environment is unused by the script.
    refresh_source("/workspace/axolotl")
    return run_remote("./cicd/cicd_mono_ep_blackwell.sh", state, full, select, False)


@app.local_entrypoint()
def main():
    run_local(cicd_mono_ep_blackwell)
