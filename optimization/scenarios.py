"""Scenario assumptions, independent of the transportation formulation."""

from decimal     import Decimal
from dataclasses import dataclass
from math        import isfinite, floor


@dataclass(frozen=True)
class TransportScenario:
    slack               : float        = 0.0
    max_interstate_share: float | None = None

    def __post_init__(self):
        if not isfinite(self.slack) or self.slack < 0:
            raise ValueError("Capacity slack must be finite and nonnegative.")

        if (
            self.max_interstate_share is not None           and
            (
                not isfinite(self.max_interstate_share)     or
                not 0 <=     self.max_interstate_share <= 1
            )
        ):
            raise ValueError(
                "Maximum interstate share must be finite and between zero and one."
            )

    def capacity(self, observed_supply: int) -> int:
        return floor(
            (Decimal(1) + Decimal(str(self.slack))) * observed_supply
        )

    def interstate_limit(self, total_admissions: int) -> int | None:
        if self.max_interstate_share is None:
            return None

        return floor(
            Decimal(str(self.max_interstate_share)) * total_admissions
        )
