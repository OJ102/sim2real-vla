"""YAML -> dataclass loader with `key=value` CLI overrides (dotted keys allowed for nested dataclasses)."""
from __future__ import annotations

import dataclasses
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _coerce(raw: str) -> Any:
    v = yaml.safe_load(raw)  # "3"->3, "true"->True, "[1,2]"->list
    if isinstance(v, str):
        try:
            return float(v)  # YAML 1.1 reads "3e-5" as a string
        except ValueError:
            pass
    return v


def _build(cls: type[T], data: dict) -> T:
    hints = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(hints)
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for k, v in data.items():
        t = hints[k].type
        if isinstance(t, type) and dataclasses.is_dataclass(t) and isinstance(v, dict):
            v = _build(t, v)
        kwargs[k] = v
    return cls(**kwargs)


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    return data


def load_config(cls: type[T], path: str | None = None, overrides: list[str] | None = None) -> T:
    data: dict = {}
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data = apply_overrides(data, overrides or [])
    return _build(cls, data)


def dump_config(cfg: Any, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f)
