import os
import sys

# Headless GL on Linux servers; macOS keeps MuJoCo's default (CGL) backend.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
