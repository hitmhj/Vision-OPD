# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.models.transformers.tiled_mlp import apply_tiled_mlp_monkey_patch
from verl.utils.device import is_torch_npu_available

# Importing a submodule does execute this package initializer first. Therefore
# every FSDP Ray worker that imports ``monkey_patch`` also receives the Ascend
# model patches before constructing Qwen. Keep CUDA/CPU environments untouched.
NPU_PATCH_LOADED = False
if is_torch_npu_available(check_device=False):
    from verl.models.transformers import npu_patch as _npu_patch  # noqa: F401

    NPU_PATCH_LOADED = True

__all__ = [
    "apply_monkey_patch",
    "apply_tiled_mlp_monkey_patch",
    "NPU_PATCH_LOADED",
]
