"""TabletopEnv: Franka Panda + 3 coloured cubes, tasks reach / lift / push, 4-D EE-delta actions at 10 Hz."""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .assets import (CAM_LOOKAT, CAM_NAME, CAM_POS, CAM_FOVY, CUBE_COLORS, CUBE_HALF, HOME_QPOS, N_LIGHTS,
                     build_model, look_at_quat)
from .ik import DLSIK
from .randomization import DRConfig, DRParams, sample_dr
from .rewards import is_success

TASKS = ("reach", "lift", "push")
ACT_SCALE = 0.03          # metres per step at |a|=1
SUBSTEPS = 50             # 50 * 2 ms = 0.1 s  (10 Hz)
MAX_STEPS = 100
WS_LO = np.array([0.25, -0.40, 0.02])
WS_HI = np.array([0.80, 0.40, 0.55])
PUSH_DIST = 0.14


class TabletopEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, task: str = "lift", split: str = "train", image_size: int = 128, render_obs: bool = True,
                 dr_config: DRConfig | None = None):
        assert task in TASKS and split in ("train", "heldout")
        self.task, self.split, self.image_size, self.render_obs = task, split, image_size, render_obs
        self.dr_config = dr_config or DRConfig()
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        nm = lambda t, n: mujoco.mj_name2id(m, t, n)  # noqa: E731
        O = mujoco.mjtObj
        self.site = nm(O.mjOBJ_SITE, "grasp")
        self.cam = nm(O.mjOBJ_CAMERA, CAM_NAME)
        self.table_geom = nm(O.mjOBJ_GEOM, "table")
        self.floor_geom = nm(O.mjOBJ_GEOM, "floor")
        self.goal_geom = nm(O.mjOBJ_GEOM, "goal")
        self.cube_body = [nm(O.mjOBJ_BODY, f"cube_{c}") for c in CUBE_COLORS]
        self.cube_geom = [nm(O.mjOBJ_GEOM, f"cube_{c}") for c in CUBE_COLORS]
        self.cube_qadr = [int(m.jnt_qposadr[m.body_jntadr[b]]) for b in self.cube_body]
        self.arm_bodies = {nm(O.mjOBJ_BODY, n) for n in
                           ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand"]}
        self.finger_geoms = [g for g in range(m.ngeom) if m.body_parentid[m.geom_bodyid[g]] ==
                             nm(O.mjOBJ_BODY, "hand")]
        self.top_light = nm(O.mjOBJ_LIGHT, "top")
        self.lights = [nm(O.mjOBJ_LIGHT, f"light{i}") for i in range(N_LIGHTS)]
        # originals for DR restore
        self._orig = {k: getattr(m, k).copy() for k in
                      ("geom_pos", "geom_rgba", "geom_friction", "geom_matid", "body_mass", "body_inertia",
                       "light_dir", "light_diffuse", "light_active", "cam_pos", "cam_quat", "cam_fovy",
                       "actuator_gainprm", "actuator_biasprm")}
        self.ik = DLSIK(m)
        self._renderer = None
        # down-pointing orientation = home orientation
        d = self.data
        d.qpos[:7] = HOME_QPOS
        mujoco.mj_forward(m, d)
        self.target_mat = d.site_xmat[self.site].reshape(3, 3).copy()

        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, (11,), np.float32)})
        self.dr: DRParams | None = None
        self.t = 0

    # ---------------------------------------------------------------- DR / layout
    def _apply_dr(self, p: DRParams):
        m, o = self.model, self._orig
        for k, v in o.items():
            getattr(m, k)[...] = v
        pos = CAM_POS + p.cam_pos_offset
        m.cam_pos[self.cam] = pos
        m.cam_quat[self.cam] = look_at_quat(pos, CAM_LOOKAT + p.cam_look_offset)
        m.cam_fovy[self.cam] = CAM_FOVY + p.cam_fov_delta
        m.light_active[self.top_light] = 0
        for i, l in enumerate(self.lights):
            m.light_active[l] = int(i < p.n_lights)
            m.light_dir[l] = p.light_dir[i]
            m.light_diffuse[l] = p.light_diffuse[i]
        m.geom_matid[self.table_geom] = p.table_mat
        m.geom_matid[self.floor_geom] = p.floor_mat
        for i, g in enumerate(self.cube_geom):
            m.geom_rgba[g, :3] = p.cube_rgb[i]
            m.geom_friction[g, 0] = p.friction
            b = self.cube_body[i]
            m.body_inertia[b] = o["body_inertia"][b] * (p.cube_mass / o["body_mass"][b])
            m.body_mass[b] = p.cube_mass
        m.geom_friction[self.table_geom, 0] = p.friction
        for g in self.finger_geoms:
            m.geom_friction[g, 0] = p.friction
        m.actuator_gainprm[:7, 0] = o["actuator_gainprm"][:7, 0] * p.kp_scale
        m.actuator_biasprm[:7, 1] = o["actuator_biasprm"][:7, 1] * p.kp_scale

    def _sample_layout(self, rng: np.random.Generator):
        for _ in range(1000):
            xy = np.stack([rng.uniform(0.40, 0.58, 3), rng.uniform(-0.22, 0.22, 3)], axis=1)
            ti = self.target_idx
            tgt = xy[ti]
            th = rng.uniform(-np.pi / 6, np.pi / 6)
            d = np.array([np.cos(th), np.sin(th)])
            goal = tgt + PUSH_DIST * d
            if not (0.35 <= goal[0] <= 0.72 and abs(goal[1]) <= 0.30):
                continue
            seg_a, seg_b = tgt - 0.09 * d, goal
            ok = True
            for j in range(3):
                if j == ti:
                    continue
                if np.linalg.norm(xy[j] - tgt) < 0.10:
                    ok = False
                if self.task == "push":
                    ab = seg_b - seg_a
                    s = np.clip(np.dot(xy[j] - seg_a, ab) / np.dot(ab, ab), 0, 1)
                    if np.linalg.norm(xy[j] - (seg_a + s * ab)) < 0.08 or np.linalg.norm(xy[j] - goal) < 0.09:
                        ok = False
            for a in range(3):
                for b in range(a + 1, 3):
                    if np.linalg.norm(xy[a] - xy[b]) < 0.08:
                        ok = False
            if ok:
                return xy, goal
        raise RuntimeError("layout sampling failed")

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed) if seed is not None else self.np_random
        self.dr = sample_dr(self.split, rng, self.dr_config)
        self._apply_dr(self.dr)
        self.target_idx = int(rng.integers(3))
        self.target_color = CUBE_COLORS[self.target_idx]
        xy, goal = self._sample_layout(rng)
        self.goal_xy = goal
        self.model.geom_pos[self.goal_geom, :2] = goal
        if self.task != "push":
            self.model.geom_rgba[self.goal_geom, 3] = 0.0  # hide zone when unused
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        mujoco.mj_setConst(m, d)  # refresh derived mass quantities (invweight0, subtreemass) after DR
        d.qpos[:7] = HOME_QPOS + rng.uniform(-0.02, 0.02, 7)
        d.qpos[7:9] = 0.04
        for i, a in enumerate(self.cube_qadr):
            d.qpos[a:a + 3] = (xy[i, 0], xy[i, 1], CUBE_HALF)
            d.qpos[a + 3:a + 7] = (1, 0, 0, 0)
        d.ctrl[:7] = d.qpos[:7]
        d.ctrl[7] = 255
        mujoco.mj_forward(m, d)
        for _ in range(100):  # settle
            mujoco.mj_step(m, d)
        self.t = 0
        self._ee_target_prev = None
        return self._obs(), self._info(reset=True)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        m, d = self.model, self.data
        ee = d.site_xpos[self.site].copy()
        tgt = np.clip(ee + a[:3] * ACT_SCALE, WS_LO, WS_HI)
        d.ctrl[:7] = self.ik.solve(d.qpos[:7], tgt, self.target_mat)
        d.ctrl[7] = 255.0 * (1 - a[3]) / 2
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        self.t += 1
        info = self._info()
        terminated = info["success"]
        truncated = self.t >= MAX_STEPS and not terminated
        return self._obs(), float(terminated), terminated, truncated, info

    # ---------------------------------------------------------------- helpers
    def privileged(self) -> dict:
        d = self.data
        return {"ee_pos": d.site_xpos[self.site].copy(), "grip_width": float(d.qpos[7] + d.qpos[8]),
                "cube_pos": np.array([d.xpos[b] for b in self.cube_body]), "target_idx": self.target_idx,
                "goal_xy": self.goal_xy.copy()}

    @property
    def instruction(self) -> str:
        c = self.target_color
        return {"reach": f"reach the {c} cube", "lift": f"lift the {c} cube",
                "push": f"push the {c} cube to the yellow zone"}[self.task]

    def _obs(self):
        d = self.data
        proprio = np.concatenate([d.site_xpos[self.site], [d.qpos[7] + d.qpos[8]], d.qpos[:7]]).astype(np.float32)
        return {"image": self.render() if self.render_obs else np.zeros((self.image_size,) * 2 + (3,), np.uint8),
                "proprio": proprio}

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)
        self._renderer.update_scene(self.data, camera=CAM_NAME)
        return self._renderer.render().copy()

    def _contacts(self) -> dict:
        m, d = self.model, self.data
        arm_table = arm_dist = 0
        dist_geoms = {g for i, g in enumerate(self.cube_geom) if i != self.target_idx}
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            for ga, gb, ba in ((g1, g2, b1), (g2, g1, b2)):
                if ba in self.arm_bodies and ga not in self.finger_geoms:
                    arm_table += gb == self.table_geom
                    arm_dist += gb in dist_geoms
        return {"arm_table": int(arm_table), "arm_distractor": int(arm_dist), "total": int(d.ncon)}

    def _info(self, reset: bool = False) -> dict:
        d = self.data
        ee = d.site_xpos[self.site].copy()
        cube = d.xpos[self.cube_body[self.target_idx]].copy()
        info = {"success": is_success(self.task, ee, cube, self.goal_xy), "qacc": d.qacc[:7].copy(),
                "qvel": d.qvel[:7].copy(), "qpos": d.qpos[:7].copy(), "contacts": self._contacts(),
                "ee_pos": ee, "instruction": self.instruction, "task": self.task, "step": self.t}
        if reset:
            info["dr"] = self.dr.to_dict()
        return info

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _demo(task: str, episodes: int, out: str, split: str = "train"):
    import imageio.v2 as imageio

    from .expert import ScriptedExpert

    Path(out).mkdir(parents=True, exist_ok=True)
    env = TabletopEnv(task=task, split=split)
    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        expert = ScriptedExpert(task, np.random.default_rng(1000 + ep))
        frames = [obs["image"]]
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(expert.act(env))
            frames.append(obs["image"])
            done = term or trunc
        path = Path(out) / f"{task}_{split}_{ep}.mp4"
        imageio.mimsave(path, [np.kron(f, np.ones((2, 2, 1), np.uint8)) for f in frames], fps=10, macro_block_size=1)
        print(f"{path} frames={len(frames)} success={info['success']} instr='{info['instruction']}' "
              f"image={obs['image'].shape}")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--task", default="lift", choices=TASKS)
    ap.add_argument("--split", default="train")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="results/demo")
    a = ap.parse_args()
    if a.demo:
        _demo(a.task, a.episodes, a.out, a.split)
