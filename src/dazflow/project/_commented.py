"""A configuration as YAML with every default written out and each field's description above it."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel

__all__ = ["commented_yaml"]


def _scalar(value: Any) -> str:
    text = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=float("inf"))
    text = text.strip()
    return text.removesuffix("\n...").removesuffix("...").strip()


def _lines(model: BaseModel, indent: int) -> list[str]:
    pad = " " * indent
    out: list[str] = []
    data = model.model_dump(mode="json")
    for name, info in type(model).model_fields.items():
        if info.exclude:
            continue
        if info.description:
            out.extend(f"{pad}# {line}".rstrip() for line in info.description.splitlines())
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            out.append(f"{pad}{name}:")
            out.extend(_lines(value, indent + 2))
        elif (
            isinstance(data[name], list)
            and data[name]
            and all(isinstance(v, dict) for v in data[name])
        ):
            out.append(f"{pad}{name}:")
            block = yaml.safe_dump(
                data[name],
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
                width=float("inf"),
            )
            out.extend(f"{pad}  {line}" for line in block.splitlines())
        else:
            out.append(f"{pad}{name}: {_scalar(data[name])}")
    return out


def commented_yaml(config: BaseModel, header: str | None = None) -> str:
    """`config` as YAML: every field in declaration order, defaults filled in, described.

    Loading the text back gives ``config.model_dump(mode="json")``. `header`
    lines are written first, as comments.
    """
    lines = [f"# {h}" if h else "#" for h in (header or "").splitlines()]
    if lines:
        lines.append("")
    lines.extend(_lines(config, 0))
    return "\n".join(lines) + "\n"
