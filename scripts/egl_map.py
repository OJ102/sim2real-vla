"""Map each EGL device id (MUJOCO_EGL_DEVICE_ID) to its GPU via the DRM render node -> PCI bus id -> nvidia-smi.

Prints a table and a recommended VLA_EGL_DEVICES (A100 ids only; excludes e.g. the DGX Display card).
Linux only. If the mapping cannot be determined it says so and prints no recommendation.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys

EGL_DRM_DEVICE_FILE_EXT = 0x3233


def egl_drm_files() -> list[str | None]:
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    q_dev = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(
        egl.eglGetProcAddress(b"eglQueryDevicesEXT"))
    q_str = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(
        egl.eglGetProcAddress(b"eglQueryDeviceStringEXT"))
    n = ctypes.c_int()
    devs = (ctypes.c_void_p * 16)()
    if not q_dev(16, devs, ctypes.byref(n)):
        raise RuntimeError("eglQueryDevicesEXT failed")
    out = []
    for i in range(n.value):
        f = q_str(devs[i], EGL_DRM_DEVICE_FILE_EXT)
        out.append(f.decode() if f else None)
    return out


def drm_pci(path: str) -> str | None:
    """/dev/dri/cardN or renderDN -> 'bus:dev.fn' (lowercase, no domain), via sysfs."""
    node = os.path.basename(path)
    link = f"/sys/class/drm/{node}/device"
    if not os.path.exists(link):
        return None
    return os.path.basename(os.path.realpath(link)).lower().split(":", 1)[-1]


def smi_gpus() -> dict[str, tuple[int, str]]:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,pci.bus_id", "--format=csv,noheader"],
                                  text=True)
    res = {}
    for line in out.strip().splitlines():
        idx, name, bus = [x.strip() for x in line.split(",")]
        res[bus.lower().split(":", 1)[-1]] = (int(idx), name)
    return res


def main() -> int:
    try:
        files, gpus = egl_drm_files(), smi_gpus()
    except Exception as e:  # noqa: BLE001
        print(f"EGL->GPU mapping: cannot determine ({type(e).__name__}: {e})")
        return 0
    good = []
    for i, f in enumerate(files):
        pci = drm_pci(f) if f else None
        gpu = gpus.get(pci) if pci else None
        desc = f"nvidia-smi index {gpu[0]} ({gpu[1]})" if gpu else "unknown"
        print(f"  EGL device {i}: drm={f} pci={pci} -> {desc}")
        if gpu and "A100" in gpu[1]:
            good.append(i)
    if good:
        print(f"recommended VLA_EGL_DEVICES={','.join(map(str, good))}  (A100 EGL ids; set in .env)")
    else:
        print("EGL->GPU mapping: could not map any EGL device to an A100; no recommendation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
