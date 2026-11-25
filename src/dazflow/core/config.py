"""Config: the pydantic base of every step and component configuration (design §8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Self

import yaml
from pydantic import ConfigDict

from dazflow.core._model import DazflowModel
from dazflow.core.errors import ConfigError

__all__ = ["Config"]

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


class _UniqueKeyLoader(yaml.SafeLoader):
    """PyYAML's safe loader, refusing a mapping that repeats a key."""


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict:
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


class Config(DazflowModel):
    """Base class of every configuration: YAML on disk, JSON Schema for tools.

    Subclasses declare fields as pydantic models do. A config is frozen,
    refuses unknown fields and refuses non-finite floats, so its YAML, its
    JSON and its schema agree. Validation errors are `ConfigError`.

    Examples
    --------
    >>> class SelectConfig(Config):
    ...     metric: str = "rmse"
    ...     tolerance: float = 0.05
    >>> SelectConfig.from_yaml("tolerance: 0.1\\n").to_yaml()
    'metric: rmse\\ntolerance: 0.1\\n'
    """

    model_config = ConfigDict(allow_inf_nan=False)

    _error: ClassVar = ConfigError
    _hint: ClassVar[str] = "Correct the configuration fields listed in 'observed'."

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        """Validate a configuration from YAML text."""
        try:
            data = yaml.load(text, Loader=_UniqueKeyLoader)  # a SafeLoader subclass
        except yaml.YAMLError as err:
            raise ConfigError(
                f"the {cls.__name__} YAML does not parse",
                observed=str(err).replace("\n", " "),
                change="Correct the YAML syntax at the position given.",
            ) from err
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError(
                f"the {cls.__name__} YAML is not a mapping",
                observed=f"a {type(data).__name__} at the top level",
                change="Write the configuration as 'field: value' lines.",
            )
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read and validate a configuration from a UTF-8 YAML file."""
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as err:
            raise ConfigError(
                f"the {cls.__name__} file cannot be read",
                observed=f"{path}: {err.strerror or err}",
                change="Check the path and that the file exists.",
            ) from err
        return cls.from_yaml(text)

    def to_yaml(self) -> str:
        """The configuration as YAML, every default filled in, fields in declaration order."""
        return yaml.safe_dump(
            self.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=float("inf"),
        )

    def dump(self, path: str | Path) -> None:
        """Write `to_yaml()` to a UTF-8 file."""
        Path(path).write_text(self.to_yaml(), encoding="utf-8", newline="\n")

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """The JSON Schema (draft 2020-12) of this configuration."""
        return {"$schema": JSON_SCHEMA_DIALECT, **cls.model_json_schema()}
