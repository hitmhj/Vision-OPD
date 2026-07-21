#!/usr/bin/env python3
"""Minimal native Ascend probe used by the unified launcher."""

import ctypes
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
    opapi = None
    try:
        opapi = ctypes.CDLL("libopapi.so")
        print(f"[npu-probe:{phase}] preloaded libopapi.so", flush=True)
    except OSError as exc:
        print(f"[WARNING] [npu-probe:{phase}] could not preload libopapi.so: {exc}", flush=True)

    torch.npu.set_device(0)
    print(f"[npu-probe:{phase}] starting one-element CPU-to-NPU copy", flush=True)
    probe = torch.empty(1, dtype=torch.float32, device="cpu").to("npu:0")
    torch.npu.synchronize()
    del probe
    del opapi
    print(f"[npu-probe:{phase}] native CPU-to-NPU copy passed", flush=True)


if __name__ == "__main__":
    main()
