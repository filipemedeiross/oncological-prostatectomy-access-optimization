"""Read the canonical CSV instance without repeating upstream preprocessing"""

import json

from pathlib     import Path
from hashlib     import sha256
from io          import BytesIO
from dataclasses import dataclass, field

import numpy  as np
import pandas as pd


@dataclass
class OptimizationData:
    origins   : tuple[str            , ...]
    providers : tuple[str            , ...]
    arcs      : tuple[tuple[str, str], ...]

    demand          : dict[str            , int  ]
    observed_supply : dict[str            , int  ]
    distance        : dict[tuple[str, str], float]

    origin_state   : dict[str, str]
    provider_state : dict[str, str]

    source_hashes          : dict[str, str      ] = field(default_factory=dict )
    destinations_by_origin : dict[str, list[str]] = field(init           =False)
    origins_by_destination : dict[str, list[str]] = field(init           =False)

    def __post_init__(self):
        self.destinations_by_origin = {i : [] for i in self.origins  }
        self.origins_by_destination = {j : [] for j in self.providers}

        for i, j in self.arcs:
            self.destinations_by_origin[i].append(j)
            self.origins_by_destination[j].append(i)


def load_instance(
    instance_dir : str | Path             ,
    period       : str = "2026_first_half",
) -> OptimizationData:
    """Load notebooks 3-4 exports; retain string codes and original float distances."""

    instance_dir = Path(instance_dir)

    names = {
        "municipalities"    : f"optimization_municipalities_{period}.csv"    ,
        "distances"         : f"optimization_distances_{period}.csv"         ,
        "instance_metadata" : f"optimization_metadata_{period}.json"         ,
        "distance_metadata" : f"optimization_distance_metadata_{period}.json",
    }

    raw    = {key : (instance_dir / name).read_bytes() for key, name  in names.items()}
    hashes = {key :  sha256(value       ).hexdigest () for key, value in raw  .items()}

    metadata          = json.loads(raw["instance_metadata"])
    distance_metadata = json.loads(raw["distance_metadata"])

    for key in (
        "municipalities"   ,
        "distances"        ,
        "instance_metadata",
    ):
        if (
            distance_metadata[f"{key}_file"  ] != names [key] or
            distance_metadata[f"{key}_sha256"] != hashes[key]
        ):
            raise ValueError(f"Stale or inconsistent instance artifact: {key}")

    if metadata   ["municipalities_file"] != names["municipalities"]:
        raise ValueError("Municipalities file differs from instance metadata.")
    if distance_metadata["distance_unit"] != "km":
        raise ValueError("Distances must be in kilometres."                   )

    municipal = pd.read_csv(
        BytesIO(
            raw["municipalities"]
        ),
        dtype={
            "municipality_code" : "string",
            "state"             : "string",
            "demand"            : "int64" ,
            "observed_supply"   : "int64" ,
        }
    ).set_index(
        "municipality_code"  ,
        verify_integrity=True,
    ).sort_index()

    distances = pd.read_csv(
        BytesIO(
            raw["distances"]
        ),
        dtype={
            "origin_code"      : "string" ,
            "destination_code" : "string" ,
            "distance_km"      : "float64",
        }
    ).sort_values(
        [
            "origin_code"     ,
            "destination_code",
        ]
    )

    if municipal.empty or municipal[
        [
            "state"          ,
            "demand"         ,
            "observed_supply",
        ]
    ].isna().any().any():
        raise ValueError("Empty or incomplete municipal instance.")

    if (
            municipal.index.isna()                              .any() or
        not municipal.index.str.fullmatch(r"[0-9]{7}")          .all() or
        not municipal[["demand", "observed_supply"]].ge(0).all().all()
    ):
        raise ValueError("Invalid municipality code or negative admission count.")

    demand = municipal.loc[municipal.demand         .gt(0), "demand"         ].to_dict()
    supply = municipal.loc[municipal.observed_supply.gt(0), "observed_supply"].to_dict()

    origins   = tuple(demand)
    providers = tuple(supply)

    keys = [
        "origin_code"     ,
        "destination_code",
    ]

    if (
        distances[keys + ["distance_km"]].isna().any().any() or
        distances.duplicated(keys)                    .any() or

        set(distances.origin_code     ) != set(origins  ) or
        set(distances.destination_code) != set(providers) or
        len(distances                 ) != len(origins  ) * len(providers)
    ):
        raise ValueError("Distance CSV must cover I x J exactly once.")

    if (
        not np.isfinite(distances.distance_km)     .all() or
                        distances.distance_km.lt(0).any()
    ):
        raise ValueError("Nonfinite or negative distance.")

    counts = {
        "n_municipalities" : len(municipal),
        "n_origins"        : len(origins  ),
        "n_providers"      : len(providers),
    }

    for key, count in counts.items():
        if (
            metadata         [key] != count or
            distance_metadata[key] != count
        ):
            raise ValueError(f"Metadata count differs: {key}")

    if (
        not origins   or
        not providers or

        distance_metadata["n_pairs"] != len(distances      ) or
        sum(demand.values()        ) != sum(supply.values()) or

        sum(demand.values()) != metadata["total_demand"         ] or
        sum(supply.values()) != metadata["total_observed_supply"] or
        sum(demand.values()) != metadata["n_admissions"         ]
    ):
        raise ValueError("Empty, unbalanced or inconsistent instance totals.")

    distance = distances.set_index(keys).distance_km.to_dict()

    return OptimizationData(
        origins  =origins        ,
        providers=providers      ,
        arcs     =tuple(distance),

        demand         =demand  ,
        observed_supply=supply  ,
        distance       =distance,

        origin_state  =municipal.loc[list(origins  ), "state"].to_dict(),
        provider_state=municipal.loc[list(providers), "state"].to_dict(),

        source_hashes={
            names[key] : value
            for key, value in hashes.items()
        },
    )
