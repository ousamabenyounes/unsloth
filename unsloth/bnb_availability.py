# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

"""One shared answer to "is bitsandbytes usable on this host?".

A successful `import bitsandbytes` is not that answer. A wheel whose native side
failed to load still imports, and what it leaves behind varies by version:

  * no `functional` at all, so reading it raises at import
  * `functional.lib is None` (0.45.5, the floor in pyproject.toml), so the ctypes
    binds raise "'NoneType' object has no attribute cdequantize_blockwise_fp32"
  * `functional.lib` missing that symbol, or an ErrorHandlerMockBNBNativeLibrary
    (0.46 onwards). These do NOT raise: BNBNativeLibrary.__getattr__ returns a
    plain `throw_on_call` closure, so the binds succeed and 4bit dies later
    inside a kernel. Hence the `restype` check below - a real handle is a ctypes
    function pointer, a deferred failure is a Python function.

device_type.py, _gpu_init.py and kernels/utils.py must agree on the answer, or
ALLOW_BITSANDBYTES stays true while the kernels fall back to the stub and
loader.py forwards a 4bit request instead of the advertised 16bit fallback.

A leaf module on purpose: it imports nothing from unsloth - device_type.py is
imported very early and would be a cycle - and takes the device type as an
argument. bitsandbytes is imported inside a function, so `import unsloth` never
hard-requires it.
"""

__all__ = [
    "bitsandbytes_symbols",
    "check_bitsandbytes",
    "probe_bitsandbytes",
]

# The ctypes handles kernels/utils.py binds at module scope. Keep in step with the
# `bnb.functional.lib.*` reads there - a test asserts the two match.
_C_SYMBOLS = (
    "cdequantize_blockwise_fp32",
    "cdequantize_blockwise_fp16_nf4",
    "cdequantize_blockwise_bf16_nf4",
)
# 4bit inference is a gemv on xpu and a naive gemm everywhere else, so probing the
# xpu names on cuda would write off a perfectly good wheel.
_C_SYMBOLS_XPU = (
    "cgemv_4bit_inference_fp16",
    "cgemv_4bit_inference_bf16",
)
_C_SYMBOLS_GEMM = (
    "cgemm_4bit_inference_naive_fp16",
    "cgemm_4bit_inference_naive_bf16",
)


def bitsandbytes_symbols(device_type):
    """Names kernels/utils.py reads off `bitsandbytes.functional.lib`."""
    tail = _C_SYMBOLS_XPU if device_type == "xpu" else _C_SYMBOLS_GEMM
    return _C_SYMBOLS + tail


def check_bitsandbytes(bnb, device_type):
    """Raise unless `bnb` can serve every module-scope read kernels/utils.py makes.

    Safe to repeat: ctypes caches the function object on the first lookup and
    bitsandbytes memoizes its wrapper, so the handles bound later are these ones.
    """
    if bnb is None:
        raise ImportError("Unsloth: `bitsandbytes` is not installed.")
    _version = bnb.__version__  # kernels/utils.py gates HAS_CUDA_STREAM on it
    functional = bnb.functional
    _get_ptr = functional.get_ptr
    lib = functional.lib  # None on a 0.45.5 native-load failure
    for symbol in bitsandbytes_symbols(device_type):
        # Only a ctypes foreign function has `restype`; a deferred failure is a
        # plain closure that raises at call time instead of here.
        if not hasattr(getattr(lib, symbol), "restype"):
            raise AttributeError(
                f"Unsloth: `bitsandbytes.functional.lib.{symbol}` is not a native "
                "handle - the bitsandbytes native library did not load."
            )


def probe_bitsandbytes(device_type):
    """The bitsandbytes module when it is usable here, else None."""
    try:
        import bitsandbytes
        check_bitsandbytes(bitsandbytes, device_type)
    except Exception:
        return None
    return bitsandbytes
