# Offline Python 3.10/aarch64 wheelhouse

Place the complete Vision-OPD Ascend wheelhouse in this directory before the
ModelArts dataset is mounted. The production launcher uses only this directory
(`pip --no-index --find-links`) and never downloads dependencies.

In addition to all direct and transitive dependencies resolved from the three
`requirements-ascend*.txt` files, this directory must contain the official
vLLM-Ascend 0.18.0 Atlas A2 wheels, including:

- `torch_npu-2.9.0.post1+git4c901a4-cp310-cp310-manylinux_2_28_aarch64.whl`
- `triton_ascend-3.2.0.dev20260322-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`
- compatible Python 3.10/aarch64 `torch`, `torchvision`, `torchaudio`, `vllm`
  and `vllm_ascend` wheels at the exact locked versions
- a `pip` wheel satisfying `pip>=23.3,<26`

Do not commit the binary wheels to Git. Upload them with the ModelArts dataset
or mount another directory and set `VOPD_LOCAL_WHEEL_DIR` to that path.
