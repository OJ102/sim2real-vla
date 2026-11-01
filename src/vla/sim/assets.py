"""Build the tabletop MJCF scene (Panda + table + 3 cubes + goal zone + camera + 3 lights) via MjSpec."""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import mujoco
import numpy as np

CUBE_HALF = 0.02
CUBE_COLORS = ("red", "green", "blue")
CUBE_BASE_RGB = {"red": (0.85, 0.1, 0.1), "green": (0.1, 0.8, 0.1), "blue": (0.1, 0.15, 0.85)}
N_LIGHTS = 3
log = logging.getLogger(__name__)
N_MATERIALS = 12  # ids 0-7 train, 8-11 held out
# DTD photo textures: one per DTD class (47 classes, sorted by name). Material id = 12 + class index.
# Classes 0-29 are train textures, classes 30-46 only appear in the `heldout_photo` split.
N_DTD = 47
PHOTO_BASE = 12
TRAIN_PHOTO_MATS = tuple(range(PHOTO_BASE, PHOTO_BASE + 30))
HELDOUT_PHOTO_MATS = tuple(range(PHOTO_BASE + 30, PHOTO_BASE + N_DTD))
TRAIN_MATS = tuple(range(8)) + TRAIN_PHOTO_MATS     # procedural 0-7 + DTD train classes
HELDOUT_MATS = tuple(range(8, 12))                  # procedural held-out (split "heldout")
TEX_SIZE = 256
N_MATERIALS_TOTAL = PHOTO_BASE + N_DTD  # procedural 12 + DTD 47
CAM_NAME = "front"
CAM_POS = np.array([1.0, 0.0, 0.45])
CAM_LOOKAT = np.array([0.45, 0.0, 0.03])
CAM_FOVY = 50.0
HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

# (kind, rgb1, rgb2): 0-7 train, 8-11 held out. Mix of checker / gradient / flat builtins.
_MATERIALS = [
    ("checker", (0.55, 0.40, 0.25), (0.45, 0.32, 0.20)),   # 0 wood-ish
    ("checker", (0.70, 0.70, 0.70), (0.55, 0.55, 0.55)),   # 1 light grey
    ("flat", (0.80, 0.75, 0.60), None),                    # 2 beige
    ("gradient", (0.60, 0.60, 0.65), (0.35, 0.35, 0.40)),  # 3 slate
    ("checker", (0.30, 0.30, 0.33), (0.22, 0.22, 0.25)),   # 4 dark grey
    ("flat", (0.55, 0.50, 0.45), None),                    # 5 taupe
    ("gradient", (0.75, 0.65, 0.50), (0.50, 0.40, 0.30)),  # 6 sand
    ("checker", (0.85, 0.85, 0.80), (0.70, 0.70, 0.65)),   # 7 cream
    ("checker", (0.20, 0.45, 0.50), (0.12, 0.30, 0.35)),   # 8 teal  (held out)
    ("gradient", (0.55, 0.30, 0.50), (0.30, 0.15, 0.30)),  # 9 plum  (held out)
    ("flat", (0.95, 0.95, 0.98), None),                    # 10 near-white (held out)
    ("checker", (0.15, 0.15, 0.40), (0.08, 0.08, 0.25)),   # 11 navy (held out)
]


def dtd_dir() -> Path:
    return Path(os.environ.get("VLA_DTD_DIR") or (Path(os.environ.get("VLA_DATA") or Path.home() / ".cache/vla") / "dtd"))


def _photo_pngs(allow_missing_dtd: bool = False) -> tuple[list[str], bool]:
    """256x256 PNG per DTD class (cached next to the DTD dir).

    Missing DTD is a hard error (it would silently corrupt the train texture set and the heldout_photo split) unless
    allow_missing_dtd=True (tiny CPU tests only), which uses seeded noise placeholders."""
    src = dtd_dir()
    classes = sorted(p.name for p in src.iterdir() if p.is_dir()) if src.is_dir() else []
    real = len(classes) >= N_DTD
    if not real:
        if not allow_missing_dtd:
            raise FileNotFoundError(f"DTD not found / incomplete at {src} ({len(classes)}/{N_DTD} classes). "
                                    "Run scripts/fetch_dtd.sh, or pass allow_missing_dtd=True (tests only).")
        warnings.warn(f"DTD not found at {src}: photo materials are NOISE placeholders (allow_missing_dtd=True)")
    out = src.parent / f"dtd_{TEX_SIZE}" if real else Path.home() / ".cache/vla/dtd_noise"
    out.mkdir(parents=True, exist_ok=True)
    log.info("texture source: %s", f"real DTD from {src}" if real else "NOISE placeholders")
    from PIL import Image

    paths = []
    for i in range(N_DTD):
        f = out / (f"{classes[i]}.png" if real else f"noise_{i:02d}.png")
        if not f.exists():
            if real:
                img = sorted((src / classes[i]).glob("*.jpg"))[0]
                Image.open(img).convert("RGB").resize((TEX_SIZE, TEX_SIZE), Image.BILINEAR).save(f)
            else:
                rgb = np.random.default_rng(i).integers(0, 255, (TEX_SIZE // 8, TEX_SIZE // 8, 3), dtype=np.uint8)
                Image.fromarray(rgb).resize((TEX_SIZE, TEX_SIZE), Image.NEAREST).save(f)
        paths.append(str(f))
    return paths, real


def build_model(allow_missing_dtd: bool = False) -> mujoco.MjModel:
    from robot_descriptions import panda_mj_description as p

    spec = mujoco.MjSpec.from_file(p.MJCF_PATH)
    spec.option.timestep = 0.002
    spec.visual.headlight.ambient = (0.2, 0.2, 0.2)
    spec.visual.headlight.diffuse = (0.2, 0.2, 0.2)
    spec.visual.headlight.specular = (0, 0, 0)
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 640

    # sky + 12 materials
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=(0.3, 0.5, 0.7), rgb2=(0.05, 0.05, 0.1), width=64, height=512)
    builtin = {"checker": mujoco.mjtBuiltin.mjBUILTIN_CHECKER, "gradient": mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
               "flat": mujoco.mjtBuiltin.mjBUILTIN_FLAT}
    for i, (kind, c1, c2) in enumerate(_MATERIALS):
        spec.add_texture(name=f"tex{i}", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=builtin[kind],
                         rgb1=c1, rgb2=c2 if c2 else c1, width=128, height=128)
        m = spec.add_material(name=f"mat{i}", texuniform=True, texrepeat=(4, 4))
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tex{i}"

    photos, _ = _photo_pngs(allow_missing_dtd)
    for i, f in enumerate(photos):
        spec.add_texture(name=f"photo{i}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=f)
        m = spec.add_material(name=f"mat{PHOTO_BASE + i}", texuniform=True, texrepeat=(2, 2))
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"photo{i}"

    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(3, 3, 0.05), pos=(0, 0, -0.75), material="mat0")
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.35, 0.5, 0.05), pos=(0.5, 0, -0.05),
                material="mat1", friction=(0.9, 0.005, 0.0001))
    wb.add_geom(name="goal", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, 0.0005), pos=(0.5, 0.2, 0.0005),
                rgba=(1.0, 0.9, 0.1, 1), contype=0, conaffinity=0)
    for i in range(N_LIGHTS):
        wb.add_light(name=f"light{i}", pos=(0.5, 0, 1.5), dir=(0, 0, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                     diffuse=(0.7, 0.7, 0.7), specular=(0.1, 0.1, 0.1), castshadow=0)
    wb.add_camera(name=CAM_NAME, pos=tuple(CAM_POS), fovy=CAM_FOVY)
    for i, c in enumerate(CUBE_COLORS):
        b = wb.add_body(name=f"cube_{c}", pos=(0.45, -0.2 + 0.2 * i, CUBE_HALF))
        b.add_freejoint()
        b.add_geom(name=f"cube_{c}", type=mujoco.mjtGeom.mjGEOM_BOX, size=(CUBE_HALF,) * 3, mass=0.1,
                   rgba=CUBE_BASE_RGB[c] + (1,), friction=(0.9, 0.005, 0.0001))

    # stiffer gripper (same open equilibrium, 8x grip force) so cubes do not slip while lifting
    grip = next(a for a in spec.actuators if a.name == "actuator8")
    grip.gainprm[0] *= 8
    grip.biasprm[1] *= 8
    grip.biasprm[2] *= 8

    # EE site between the fingertip pads
    hand = next(b for b in spec.bodies if b.name == "hand")
    hand.add_site(name="grasp", pos=(0, 0, 0.1034), size=(0.005,) * 3, rgba=(1, 0, 1, 0.0))
    return spec.compile()


def look_at_quat(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera quaternion (MuJoCo: looks along -z, y up) for a camera at pos looking at target."""
    z = pos - target
    z = z / np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).ravel())
    return q
