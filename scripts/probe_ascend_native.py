#!/usr/bin/env python3
"""Minimal native Ascend probe used by the unified launcher."""

import sys


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "unspecified"
    print(f"[npu-probe:{phase}] importing torch", flush=True)

    import torch

    abi = getattr(torch, "compiled_with_cxx11_abi", lambda: getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "unknown"))()
    print(
        f"[npu-probe:{phase}] torch={torch.__version__} "
        f"cxx11_abi={abi} path={torch.__file__}",
        flush=True,
    )

    import torch_npu

    print(
        f"[npu-probe:{phase}] torch_npu={torch_npu.__version__} path={torch_npu.__file__}",
        flush=True,
    )
    # Do not dlopen libopapi.so manually here.  torch-npu loads the required
    # operator libraries through its own runtime path; forcing ctypes.CDLL can
    # itself enter the failing native loader path before set_device/copy and
    # hides which real torch-npu operation triggers the fault.
    print(f"[npu-probe:{phase}] setting logical device 0", flush=True)
    torch.npu.set_device(0)
    print(f"[npu-probe:{phase}] logical device 0 selected", flush=True)
    print(f"[npu-probe:{phase}] starting one-element CPU-to-NPU copy", flush=True)
    probe = torch.empty(1, dtype=torch.float32, device="cpu").to("npu:0")
    torch.npu.synchronize()
    del probe
    print(f"[npu-probe:{phase}] native CPU-to-NPU copy passed", flush=True)


if __name__ == "__main__":
    main()
