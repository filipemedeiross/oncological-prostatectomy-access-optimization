"""Build or read integer-cost instances with UF from IBGE municipality prefixes"""

import numpy  as np

from pathlib     import Path
from hashlib     import sha256
from dataclasses import dataclass, field

from src.reallocation import RESOLVED_STATUSES, _validate_routes


@dataclass
class OptimizationData:
    origins   : tuple[str, ...]
    providers : tuple[str, ...]
    arcs      : tuple[tuple[str, str], ...]

    origin_state    : dict[str, str]
    provider_state  : dict[str, str]
    demand          : dict[str, int]
    observed_supply : dict[str, int]
    distance        : dict[tuple[str, str], int]

    source_hashes          : dict[str, str      ] = field(default_factory=dict )
    destinations_by_origin : dict[str, list[str]] = field(init           =False)
    origins_by_destination : dict[str, list[str]] = field(init           =False)

    def __post_init__(self):
        self.destinations_by_origin = {i : [] for i in self.origins  }
        self.origins_by_destination = {j : [] for j in self.providers}

        for i, j in self.arcs:
            self.destinations_by_origin[i].append(j)
            self.origins_by_destination[j].append(i)


def build_instance(od, municipal=None) -> OptimizationData:
    """
    Use the prepared complete OD, with the same rounding as the TXT export.

    Original decimal distances remain in ``od`` for evaluation. No TXT file or
    new distance calculation is needed to build an annual historical instance.
    """

    _validate_routes(od)

    if not od.route_status.isin(RESOLVED_STATUSES).all():
        missing = od.loc[~od.route_status.isin(RESOLVED_STATUSES)]

        raise ValueError(
            f"Complete model distances required: {len(missing)} unresolved OD pairs "
            f"with {int(missing.observed_flow.sum())} observed admissions."
        )

    demand = od.groupby("origin_code"     ).observed_flow.sum().sort_index()
    supply = od.groupby("destination_code").observed_flow.sum().sort_index()

    if demand.le(0).any() or supply.le(0).any():
        raise ValueError("Every origin/provider must have positive observed demand/production.")
    if len(od) != len(demand) * len(supply):
        raise ValueError("Model OD must contain the full origin x observed-provider product.")

    if municipal is not None:
        counts = municipal.set_index("municipality_code", verify_integrity=True)

        for actual, column in [
            (demand, "demand"         ),
            (supply, "observed_supply"),
        ]:
            expected = counts.loc[counts[column].gt(0), column].sort_index()

            if set(actual.index) != set(expected.index) or not actual.eq(expected.reindex(actual.index)).all():
                raise ValueError(f"Model OD differs from municipal {column}.")

    table  = od.sort_values(["origin_code", "destination_code"])
    values = np.rint(table.distance_km.to_numpy(dtype=float))

    if not np.isfinite(values).all() or (values < 0).any() or (values > np.iinfo(np.uint16).max).any():
        raise ValueError("Rounded model distances must fit the unsigned 16-bit TXT format.")

    arcs = tuple(
        table[
            [
                "origin_code"     ,
                "destination_code",
            ]
        ].itertuples(
            index=False,
            name =None ,
        )
    )

    return OptimizationData(
        origins        =tuple(demand.index),
        providers      =tuple(supply.index),
        arcs           =arcs,

        origin_state   ={i: i[:2] for i in demand.index},
        provider_state ={j: j[:2] for j in supply.index},

        demand         =demand.astype(int).to_dict(),
        observed_supply=supply.astype(int).to_dict(),
        distance       =dict(zip(arcs, map(int, values))),
    )


def load_instance(
    data_dir : str | Path             ,
    period   : str = "2026_first_half",
) -> OptimizationData:
    """Load integer model inputs and UF prefixes from the compact instance alone"""

    data_dir      = Path(data_dir)
    instance_name = f"telesurgery_{period}.txt"

    raw = (data_dir / instance_name).read_bytes()

    _, origin_block, provider_block, distance_block = (
        raw.decode("ascii").strip().split("\n\n")
    )

    demand = {code : int(value) for code, value in map(str.split, origin_block  .splitlines()[1:])}
    supply = {code : int(value) for code, value in map(str.split, provider_block.splitlines()[1:])}

    origins, providers = tuple(demand), tuple(supply)

    distance = {
        (i, j) : int(value)
        for i, row   in zip(origins  , distance_block.splitlines()[1:])
        for j, value in zip(providers, row           .split     ()    )
    }

    return OptimizationData(
        origins        =origins  ,
        providers      =providers,
        demand         =demand   ,
        observed_supply=supply   ,
        arcs           =tuple(distance),
        distance       =distance       ,

        origin_state  ={i : i[:2] for i in origins  },
        provider_state={j : j[:2] for j in providers},
        source_hashes ={
            instance_name : sha256(raw).hexdigest()
        },
    )
