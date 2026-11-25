"""`Design`: how a candidate's target and drivers are built from the series in levels.

Build and select share it, so the matrices a candidate was fitted on and
the ones it is evaluated and forecast on come from one function (design
§7.4, §7.6). A design is data: the target, the drivers with their lags,
each series' resolved transform chain, whether a linear trend term is
added, and the sample.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import Field

from dazflow import transforms
from dazflow.core._model import DazflowModel
from dazflow.core.errors import ModelError

__all__ = ["TREND", "Design"]

# The column a trend term takes in the drivers: 1 at the first period of the levels.
TREND = "trend"


class Design(DazflowModel):
    """A candidate's data: target and drivers transformed, drivers lagged, a trend, a sample.

    `chains` holds each series' resolved transform chain (as explore
    recorded it in `SeriesMeta.transforms`), applied to the levels by
    `dazflow.transforms.forward`; a series without one is used as it is.
    `lags[d]` shifts driver `d` by that many periods. With `trend`, a
    column ``trend`` counts periods from `trend_start` (1 there), so it
    extends past the sample unchanged. `start` and `end` are the sample's
    first and last periods.
    """

    _error: ClassVar = ModelError
    _hint: ClassVar[str] = "Designs are made by the build step."

    target: str
    drivers: tuple[str, ...] = ()
    lags: dict[str, int] = Field(default_factory=dict)
    chains: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    trend: bool = False
    trend_start: str | None = None
    start: str
    end: str

    @property
    def columns(self) -> tuple[str, ...]:
        """The driver columns a model is fitted on: the drivers, then ``trend``."""
        return (*self.drivers, TREND) if self.trend else self.drivers

    def _series(self, levels: pd.DataFrame, name: str) -> pd.Series:
        if name not in levels.columns:
            raise ModelError(
                f"the levels have no series {name!r}",
                observed=f"columns: {', '.join(map(str, levels.columns))}",
                change="Pass the series in levels that the candidate was built on.",
            )
        x = levels[name].astype("float64").rename(name)
        chain = self.chains.get(name, ())
        return transforms.forward(x, list(chain))[0] if chain else x

    def _trend(self, index: pd.PeriodIndex) -> pd.Series:
        first = pd.Period(self.trend_start, freq=index.freq)
        return pd.Series(
            np.array([(p - first).n + 1 for p in index], dtype="float64"), index=index, name=TREND
        )

    def frames(self, levels: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame | None]:
        """The target and drivers on the sample, as `fit` takes them (``X`` None without drivers).

        A missing value inside the sample is a `ModelError`: the levels are
        not those the design was made from.
        """
        index = levels.index
        sample = index[(index >= pd.Period(self.start, freq=index.freq))]
        sample = sample[sample <= pd.Period(self.end, freq=index.freq)]
        y = self._series(levels, self.target).loc[sample]
        cols = {
            d: self._series(levels, d).shift(self.lags.get(d, 0)).loc[sample] for d in self.drivers
        }
        if self.trend:
            cols[TREND] = self._trend(sample)
        X = pd.DataFrame(cols, index=sample) if cols else None
        frame = y.to_frame() if X is None else pd.concat([y.to_frame(), X], axis=1)
        if len(sample) == 0 or frame.isna().any().any():
            raise ModelError(
                "the levels do not cover the design's sample",
                observed=f"sample {self.start}..{self.end}; "
                f"{int(frame.isna().any(axis=1).sum())} periods with a missing value",
                change="Pass the series in levels the candidate was built on.",
            )
        return y, X

    def future(self, levels: pd.DataFrame, paths: pd.DataFrame, h: int) -> pd.DataFrame | None:
        """The drivers for the `h` periods after the sample, from their future `paths` in levels.

        Each driver's levels up to the sample's end are followed by its
        values in `paths` (indexed by period), transformed by its chain and
        lagged; the trend continues. ``None`` without drivers.
        """
        if not self.columns:
            return None
        freq = levels.index.freq
        end = pd.Period(self.end, freq=freq)
        periods = pd.period_range(end + 1, periods=h, freq=freq)
        cols = {}
        for d in self.drivers:
            if d not in paths.columns:
                raise ModelError(
                    f"the future paths have no driver {d!r}",
                    observed=f"columns: {', '.join(map(str, paths.columns))}",
                    change="Give a path for every driver of the candidate.",
                )
            ahead = paths[d].astype("float64").reindex(periods)
            if ahead.isna().any():
                missing = ahead.index[ahead.isna()][0]
                raise ModelError(
                    f"the future path of {d!r} does not cover the horizon",
                    observed=f"no value at {missing}; horizon {periods[0]}..{periods[-1]}",
                    change="Give every driver a value for each period after the sample.",
                )
            past = levels[d].astype("float64").loc[:end]
            full = pd.concat([past, ahead]).rename(d)
            chain = self.chains.get(d, ())
            moved = transforms.forward(full, list(chain))[0] if chain else full
            cols[d] = moved.shift(self.lags.get(d, 0)).loc[periods]
        if self.trend:
            cols[TREND] = self._trend(periods)
        return pd.DataFrame(cols, index=periods)
