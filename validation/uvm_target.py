"""Pure driver-API process holding one ordinary and one managed (UVM) allocation with known patterns."""
import ctypes, os, sys, time
cu = ctypes.CDLL("libcuda.so.1")
def ck(rc, what):
    if rc != 0:
        sys.exit(f"{what} failed rc={rc}")
log, stop = sys.argv[1], sys.argv[2]
managed = sys.argv[3:4] != ["plain"]  # "plain" skips the managed allocation (a control)
N = 64 * 2 ** 20  # 64 Mi words = 256 MB each
ck(cu.cuInit(0), "cuInit")
dev = ctypes.c_int(); ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ctx = ctypes.c_void_p(); ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "retain")
ck(cu.cuCtxSetCurrent(ctx), "setcurrent")
d = ctypes.c_uint64(); ck(cu.cuMemAlloc_v2(ctypes.byref(d), ctypes.c_size_t(N * 4)), "cuMemAlloc")
m = ctypes.c_uint64()
if managed:
    ck(cu.cuMemAllocManaged(ctypes.byref(m), ctypes.c_size_t(N * 4), ctypes.c_uint(1)), "cuMemAllocManaged")
ck(cu.cuMemsetD32_v2(d, ctypes.c_uint(0xABCD1234), ctypes.c_size_t(N)), "memset d")
if managed:
    ck(cu.cuMemsetD32_v2(m, ctypes.c_uint(0x5A5A5A5A), ctypes.c_size_t(N)), "memset m")
ck(cu.cuCtxSynchronize(), "sync")
f = open(log, "w", buffering=1)
i = 0
while not os.path.exists(stop):
    rc = cu.cuCtxSynchronize()
    f.write(f"{i} sync_rc={rc} {time.time():.3f}\n"); i += 1
    time.sleep(0.05)
host = (ctypes.c_uint32 * N)()
rc = cu.cuMemcpyDtoH_v2(host, d, ctypes.c_size_t(N * 4))
ok_d = rc == 0 and all(host[j] == 0xABCD1234 for j in range(0, N, 4099))
ok_m = all(v == 0x5A5A5A5A for v in (ctypes.c_uint32 * N).from_address(m.value)[::4099]) if managed else "n/a"
f.write(f"END copy_rc={rc} device_ok={ok_d} managed_ok={ok_m}\n")
