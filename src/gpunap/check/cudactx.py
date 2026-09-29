"""A minimal CUDA driver-API context for target processes, through ctypes. No PyTorch.

GPU buffers are filled chunk by chunk with a 32-bit value that depends on the seed and the chunk,
and verified by reading back three samples from every chunk, so a chunk restored to the wrong place
or not restored at all shows up.
"""
from __future__ import annotations

import ctypes
from typing import List, Optional, Tuple

CHUNK = 64 * 2 ** 20
SAMPLE = 4096


class CudaError(RuntimeError):
    pass


def pattern_value(seed: int, chunk: int) -> int:
    return ((seed * 1000003 + (chunk + 1) * 2654435761) & 0xFFFFFFFF) | 1


def chunk_plan(nbytes: int, chunk: int = CHUNK) -> List[Tuple[int, int]]:
    """(byte offset, 32-bit word count) for each chunk of a buffer."""
    return [(off, min(chunk, nbytes - off) // 4) for off in range(0, nbytes, chunk)]


def sample_offsets(chunk_off: int, chunk_bytes: int, sample: int = SAMPLE) -> List[int]:
    if chunk_bytes <= sample:
        return [chunk_off]
    mid = chunk_off + (chunk_bytes // 2 // 4) * 4
    return sorted({chunk_off, mid, chunk_off + chunk_bytes - sample})


def check_words(data: bytes, value: int) -> Optional[str]:
    """None if every 32-bit word equals value, else a description of the first mismatch."""
    want = value.to_bytes(4, "little")
    for i in range(0, len(data), 4):
        if data[i:i + 4] != want:
            return f"word {i // 4} is {int.from_bytes(data[i:i + 4], 'little'):#x}, expected {value:#x}"
    return None


class Ctx:
    def __init__(self, lib=None):
        self.lib = lib or ctypes.CDLL("libcuda.so.1")
        self.ck(self.lib.cuInit(0), "cuInit")
        dev, ctx = ctypes.c_int(), ctypes.c_void_p()
        self.ck(self.lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        self.ck(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
        self.ck(self.lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        self.ctx = ctx

    @staticmethod
    def ck(rc: int, what: str) -> None:
        if rc != 0:
            raise CudaError(f"{what} failed rc={rc}")

    def alloc(self, nbytes: int) -> int:
        p = ctypes.c_uint64()
        self.ck(self.lib.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(nbytes)), "cuMemAlloc")
        return p.value

    def alloc_managed(self, nbytes: int) -> int:
        p = ctypes.c_uint64()
        self.ck(self.lib.cuMemAllocManaged(ctypes.byref(p), ctypes.c_size_t(nbytes), ctypes.c_uint(1)),
                "cuMemAllocManaged")
        return p.value

    def free(self, ptr: int) -> int:
        return self.lib.cuMemFree_v2(ctypes.c_uint64(ptr))

    def stream(self) -> ctypes.c_void_p:
        s = ctypes.c_void_p()
        self.ck(self.lib.cuStreamCreate(ctypes.byref(s), 0), "cuStreamCreate")
        return s

    def fill(self, ptr: int, nbytes: int, seed: int) -> None:
        for i, (off, words) in enumerate(chunk_plan(nbytes)):
            self.ck(self.lib.cuMemsetD32_v2(ctypes.c_uint64(ptr + off), ctypes.c_uint(pattern_value(seed, i)),
                                            ctypes.c_size_t(words)), "cuMemsetD32")
        self.ck(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def read(self, ptr: int, nbytes: int) -> Tuple[int, bytes]:
        buf = (ctypes.c_char * nbytes)()
        rc = self.lib.cuMemcpyDtoH_v2(buf, ctypes.c_uint64(ptr), ctypes.c_size_t(nbytes))
        return rc, bytes(buf)

    def verify(self, ptr: int, nbytes: int, seed: int) -> Optional[str]:
        """None if the buffer still holds its pattern, else what is wrong."""
        for i, (off, words) in enumerate(chunk_plan(nbytes)):
            size = min(SAMPLE, words * 4)
            for s in sample_offsets(off, words * 4, SAMPLE):
                rc, data = self.read(ptr + s, size)
                if rc != 0:
                    return f"cuMemcpyDtoH rc={rc} at offset {s}"
                bad = check_words(data, pattern_value(seed, i))
                if bad:
                    return f"chunk {i} offset {s}: {bad}"
        return None

    def sync(self) -> int:
        return self.lib.cuCtxSynchronize()

    def stream_sync(self, s) -> int:
        return self.lib.cuStreamSynchronize(s)

    def event_sync(self, s) -> int:
        ev = ctypes.c_void_p()
        self.ck(self.lib.cuEventCreate(ctypes.byref(ev), 0), "cuEventCreate")
        self.ck(self.lib.cuEventRecord(ev, s), "cuEventRecord")
        return self.lib.cuEventSynchronize(ev)

    def memset_async(self, ptr: int, value: int, words: int, s) -> int:
        return self.lib.cuMemsetD32Async(ctypes.c_uint64(ptr), ctypes.c_uint(value), ctypes.c_size_t(words), s)
