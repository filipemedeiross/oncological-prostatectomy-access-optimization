"""Read the compact instance and derive UF from IBGE municipality prefixes"""

from pathlib     import Path
from hashlib     import sha256
from dataclasses import dataclass, field


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
