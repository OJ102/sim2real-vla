"""Domain randomization: DRConfig (train / heldout ranges), sample_dr(split, rng) -> DRParams.

Held-out ranges are strictly outside the train ranges (margin EPS) or wider, see PLAN.md.
`DRConfig.in_train_range(params)` tells whether a sample lies inside the train ranges.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from .assets import CUBE_BASE_RGB, CUBE_COLORS, HELDOUT_MATS, HELDOUT_PHOTO_MATS, N_LIGHTS, TRAIN_MATS

SPLITS = ("train", "heldout", "heldout_photo")

EPS = 1e-3


@dataclass
class DRConfig:
    # camera position offset per axis (m): train |d|<=0.03; heldout 0.03<|d|<=0.06
    cam_pos: float = 0.03
    cam_pos_held: float = 0.06
    # look-at offset (m): train |d|<=0.02; heldout 0.02<|d|<=0.04
    cam_look: float = 0.02
    cam_look_held: float = 0.04
    # fov delta (deg): train |d|<=4; heldout 4<|d|<=8
    cam_fov: float = 4.0
    cam_fov_held: float = 8.0
    # light diffuse intensity
    diffuse_train: tuple = (0.4, 1.0)
    diffuse_held: tuple = ((0.15, 0.4), (1.0, 1.4))  # union of two intervals
    n_lights: tuple = (1, 3)
    # sliding friction
    friction_train: tuple = (0.6, 1.2)
    friction_held: tuple = ((0.3, 0.6), (1.2, 1.8))
    # cube mass (kg)
    mass_train: tuple = (0.05, 0.2)
    mass_held: tuple = ((0.2, 0.4),)
    # light direction: source azimuth from the camera/front axis (deg), elevation (deg)
    light_az: float = 60.0                       # train |az| <= 60
    light_az_held: tuple = (90.0, 180.0)         # heldout side/back light: 90 < |az| <= 180
    light_el: tuple = (45.0, 75.0)               # train elevation
    light_el_held: tuple = (20.0, 40.0)          # heldout low light (az then within train range)
    kp_jitter: float = 0.10                      # train: +-10% actuator kp
    kp_held: tuple = (0.10, 0.25)                # heldout: 10 < |dkp| <= 25 %
    cube_color_jitter: float = 0.12

    def in_train_range(self, p: "DRParams") -> bool:
        ok = np.all(np.abs(p.cam_pos_offset) <= self.cam_pos) and np.all(np.abs(p.cam_look_offset) <= self.cam_look)
        ok = ok and abs(p.cam_fov_delta) <= self.cam_fov
        lo, hi = self.diffuse_train
        ok = ok and np.all((p.light_diffuse[: p.n_lights] >= lo) & (p.light_diffuse[: p.n_lights] <= hi))
        az, el = np.degrees(p.light_az[: p.n_lights]), np.degrees(p.light_el[: p.n_lights])
        ok = ok and np.all(np.abs(az) <= self.light_az) and np.all((el >= self.light_el[0]) & (el <= self.light_el[1]))
        ok = ok and np.all(np.abs(p.kp_scale - 1) <= self.kp_jitter)
        ok = ok and self.friction_train[0] <= p.friction <= self.friction_train[1]
        ok = ok and self.mass_train[0] <= p.cube_mass <= self.mass_train[1]
        ok = ok and p.table_mat in TRAIN_MATS and p.floor_mat in TRAIN_MATS
        return bool(ok)


@dataclass
class DRParams:
    cam_pos_offset: np.ndarray       # (3,) m
    cam_look_offset: np.ndarray      # (3,) m
    cam_fov_delta: float             # deg
    n_lights: int
    light_dir: np.ndarray            # (N_LIGHTS, 3) unit, direction light travels (derived from az/el)
    light_az: np.ndarray             # (N_LIGHTS,) rad, source azimuth from the camera/front axis
    light_el: np.ndarray             # (N_LIGHTS,) rad, source elevation
    light_diffuse: np.ndarray        # (N_LIGHTS,) scalar intensity
    table_mat: int
    floor_mat: int
    cube_rgb: np.ndarray             # (3 cubes in CUBE_COLORS order, 3)
    friction: float
    cube_mass: float
    kp_scale: np.ndarray             # (7,) per-arm-actuator scale

    def to_dict(self) -> dict:
        return {f.name: (getattr(self, f.name).tolist() if isinstance(getattr(self, f.name), np.ndarray)
                         else getattr(self, f.name)) for f in fields(self)}


def _signed_band(rng, lo, hi, size):
    """|x| in (lo, hi], random sign."""
    mag = rng.uniform(lo + EPS, hi, size=size)
    return mag * rng.choice([-1.0, 1.0], size=size)


def _uniform_union(rng, intervals):
    # pick an interval proportional to its length, shrink lower edge by EPS so it stays outside train
    lens = np.array([b - a for a, b in intervals])
    a, b = intervals[rng.choice(len(intervals), p=lens / lens.sum())]
    return rng.uniform(a + EPS, b - EPS)


def sample_dr(split: str, rng: np.random.Generator, cfg: DRConfig | None = None) -> DRParams:
    cfg = cfg or DRConfig()
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split}")
    held = split != "train"
    if held:
        cam_pos = _signed_band(rng, cfg.cam_pos, cfg.cam_pos_held, 3)
        cam_look = _signed_band(rng, cfg.cam_look, cfg.cam_look_held, 3)
        fov = float(_signed_band(rng, cfg.cam_fov, cfg.cam_fov_held, 1)[0])
        diffuse = np.array([_uniform_union(rng, cfg.diffuse_held) for _ in range(N_LIGHTS)])
        friction = _uniform_union(rng, cfg.friction_held)
        mass = _uniform_union(rng, cfg.mass_held)
        mats = HELDOUT_PHOTO_MATS if split == "heldout_photo" else HELDOUT_MATS
        side = rng.random(N_LIGHTS) < 0.5   # per light: side/back azimuth, else low elevation
        az = np.where(side, _signed_band(rng, cfg.light_az_held[0], cfg.light_az_held[1], N_LIGHTS),
                      rng.uniform(-cfg.light_az, cfg.light_az, N_LIGHTS))
        el = np.where(side, rng.uniform(*cfg.light_el, N_LIGHTS), rng.uniform(*cfg.light_el_held, N_LIGHTS))
        kp = _signed_band(rng, cfg.kp_held[0], cfg.kp_held[1], 7) + 1
    else:
        cam_pos = rng.uniform(-cfg.cam_pos, cfg.cam_pos, 3)
        cam_look = rng.uniform(-cfg.cam_look, cfg.cam_look, 3)
        fov = float(rng.uniform(-cfg.cam_fov, cfg.cam_fov))
        diffuse = rng.uniform(*cfg.diffuse_train, size=N_LIGHTS)
        friction = float(rng.uniform(*cfg.friction_train))
        mass = float(rng.uniform(*cfg.mass_train))
        mats = TRAIN_MATS
        az = rng.uniform(-cfg.light_az, cfg.light_az, N_LIGHTS)
        el = rng.uniform(*cfg.light_el, N_LIGHTS)
        kp = rng.uniform(1 - cfg.kp_jitter, 1 + cfg.kp_jitter, 7)
    n_lights = int(rng.integers(cfg.n_lights[0], cfg.n_lights[1] + 1))
    az, el = np.radians(az), np.radians(el)
    d = -np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    table_mat, floor_mat = (int(x) for x in rng.choice(mats, size=2, replace=False))
    rgb = []
    for c in CUBE_COLORS:
        base = np.array(CUBE_BASE_RGB[c])
        j = base + rng.uniform(-cfg.cube_color_jitter, cfg.cube_color_jitter, 3)
        dom = int(np.argmax(base))
        others = [i for i in range(3) if i != dom]
        j[dom] = np.clip(j[dom], 0.65, 1.0)      # stays recognisably red/green/blue
        j[others] = np.clip(j[others], 0.0, 0.3)
        rgb.append(j)
    return DRParams(cam_pos, cam_look, fov, n_lights, d, az, el, diffuse, table_mat, floor_mat, np.array(rgb),
                    float(friction), float(mass), kp)
