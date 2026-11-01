"""EGL device helpers (Linux). Map each EGL device id (MUJOCO_EGL_DEVICE_ID) to its GPU via
DRM render node -> PCI bus id -> nvidia-smi, recommend A100-only ids, and check GL_RENDERER.

    python -m vla.sim.egl        # prints the table + recommended VLA_EGL_DEVICES
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


def map_devices() -> list[dict]:
    """[{egl_id, drm, pci, gpu_index, gpu_name}] for every EGL device (gpu_* None if unmapped). Raises on failure."""
    files, gpus = egl_drm_files(), smi_gpus()
    rows = []
    for i, f in enumerate(files):
        pci = drm_pci(f) if f else None
        gpu = gpus.get(pci) if pci else None
        rows.append({"egl_id": i, "drm": f, "pci": pci, "gpu_index": gpu[0] if gpu else None,
                     "gpu_name": gpu[1] if gpu else None})
    return rows


def recommended_ids() -> list[int]:
    """EGL ids that map to an A100 ([] if the mapping cannot be determined)."""
    try:
        return [r["egl_id"] for r in map_devices() if r["gpu_name"] and "A100" in r["gpu_name"]]
    except Exception:  # noqa: BLE001
        return []


def gl_renderer() -> str:
    """GL_RENDERER of the current context (call after a render so the EGL context is current)."""
    from OpenGL import GL

    r = GL.glGetString(GL.GL_RENDERER)
    return r.decode() if isinstance(r, bytes) else str(r)


def assert_nvidia_renderer(env) -> str:
    """Render once and require an NVIDIA GL renderer (software renderers like llvmpipe must fail loudly).
    Skipped (returns the renderer string) off Linux, where MuJoCo uses the platform default."""
    import sys

    env.render()
    r = gl_renderer()
    if sys.platform.startswith("linux") and "NVIDIA" not in r:
        raise RuntimeError(f"GL_RENDERER={r!r} is not NVIDIA: refusing to render on a software/other device "
                           f"(MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')})")
    return r


def main() -> int:
    try:
        rows = map_devices()
    except Exception as e:  # noqa: BLE001
        print(f"EGL->GPU mapping: cannot determine ({type(e).__name__}: {e})")
        return 0
    for r in rows:
        desc = f"nvidia-smi index {r['gpu_index']} ({r['gpu_name']})" if r["gpu_name"] else "unknown"
        print(f"  EGL device {r['egl_id']}: drm={r['drm']} pci={r['pci']} -> {desc}")
    good = [r["egl_id"] for r in rows if r["gpu_name"] and "A100" in r["gpu_name"]]
    if good:
        print(f"recommended VLA_EGL_DEVICES={','.join(map(str, good))}  (A100 EGL ids; set in .env)")
    else:
        print("EGL->GPU mapping: could not map any EGL device to an A100; no recommendation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
