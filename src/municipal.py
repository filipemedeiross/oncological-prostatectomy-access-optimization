"""
Observed municipal counts, IBGE centroid locations and OD validation.

No admissions are filtered or deduplicated here. Coordinates describe municipal
geometry centroids, while production counts describe observed care, not capacity.
"""

import re
import gzip
import json
import httpx

from pathlib  import Path
from tempfile import NamedTemporaryFile

import numpy  as np
import pandas as pd


ADMISSION_COLUMNS = {
    "aih_number"     : "admission_id"  ,
    "cnes"           : "hospital_id"   ,
    "procedure_code" : "procedure_code",
    "admission_date" : "admission_date",

    "residence_ibge_code"    : "residence_code",
    "residence_municipality" : "residence_name",
    "procedure_ibge_code"    : "provider_code" ,
    "procedure_municipality" : "provider_name" ,
}

COUNT_COLUMNS     = [
    "demand"         ,
    "observed_supply",
    "local_care"     ,
    "hospitals"      ,
]
MUNICIPAL_COLUMNS = [
    "municipality_code",
    "municipality"     ,
    "state_code"       ,
    "state"            ,
    "latitude"         ,
    "longitude"        ,

    *COUNT_COLUMNS,
]
OD_COLUMNS        = [
    "origin_code"     ,
    "destination_code",
    "distance_km"     ,
    "travel_time_min" ,
    "observed_flow"   ,
    "same_state"      ,
    "route_status"    ,
]

STATE_CODES = dict(
    item.split(":")
    for item in """
        11:RO 12:AC 13:AM 14:RR 15:PA 16:AP 17:TO 21:MA 22:PI
        23:CE 24:RN 25:PB 26:PE 27:AL 28:SE 29:BA 31:MG 32:ES
        33:RJ 35:SP 41:PR 42:SC 43:RS 50:MS 51:MT 52:GO 53:DF
    """.split()
)


def read_admissions(
    path,

    start    ="2026-01-01",
    end      ="2026-06-30",
    procedure="0416010121",
):
    """Read CSV strictly, preserving AIH, CNES and IBGE strings"""

    admissions = pd.read_csv(
        path,

        usecols=list(ADMISSION_COLUMNS),
        dtype  ="string"               ,
    )

    admissions = admissions.rename(columns=ADMISSION_COLUMNS)

    if admissions.empty or admissions.isna().any().any():
        raise ValueError("Empty or incomplete admissions input.")

    if admissions.apply(lambda col: col.str.strip().eq("").any()).any():
        raise ValueError("Blank admissions field.")

    if admissions.duplicated(["admission_id", "hospital_id"]).any():
        raise ValueError("Repeated admission and hospital identifiers.")

    for field, length in [
        ("admission_id"  , 13), ("hospital_id"  , 7),
        ("residence_code", 7 ), ("provider_code", 7),
    ]:
        if not admissions[field].str.fullmatch(rf"[0-9]{{{length}}}").all():
            raise ValueError(f"Invalid identifier: {field}.")

    if not admissions["procedure_code"].eq(procedure).all():
        raise ValueError("Unexpected procedure in admissions input.")

    admissions["admission_date"] = pd.to_datetime(
        admissions["admission_date"],

        format="%Y-%m-%d",
        errors="raise"   ,
    )

    if not admissions["admission_date"].between(start, end).all():
        raise ValueError("Admission outside the study period.")

    return admissions


def aggregate_municipal(admissions):
    """Count both margins and actual local care over the union of municipalities"""

    reference = pd.concat(
        [
            admissions[[f"{side}_code", f"{side}_name"]].set_axis(
                [
                    "municipality_code",
                    "municipality"     ,
                ],
                axis=1
            )
            for side in ["residence", "provider"]
        ]
    ).drop_duplicates()

    reference = reference.set_index   ("municipality_code", verify_integrity=True)
    codes     = reference.index.astype("string")

    if not codes.str.fullmatch(r"[0-9]{7}").all():
        raise ValueError("Invalid municipality code.")

    local  = admissions["residence_code"].eq(admissions["provider_code"])

    counts = pd.concat(
        [
            admissions.groupby("residence_code").size().rename("demand"         ),
            admissions.groupby("provider_code" ).size().rename("observed_supply"),

            admissions.loc[local].groupby("residence_code")               .size   ().rename("local_care"),
            admissions           .groupby("provider_code" )["hospital_id"].nunique().rename("hospitals" ),
        ],
        axis=1
    ).fillna(0).astype("int64")

    municipal = reference.join(counts).sort_index().reset_index()

    municipal.insert(2, "state_code", municipal["municipality_code"].str[:2]                          )
    municipal.insert(3, "state"     , municipal["state_code"       ].map(STATE_CODES).astype("string"))

    if municipal.isna().any().any():
        raise ValueError("Municipality without a recognized UF or count.")

    if not municipal[["demand", "observed_supply"]].sum().eq(len(admissions)).all():
        raise ValueError("Municipal totals differ from admissions.")

    if municipal["local_care"].gt(municipal[["demand", "observed_supply"]].min(axis=1)).any():
        raise ValueError("Local care exceeds a municipal margin.")

    return municipal


def load_boundaries(root, level="municipio"):
    """Reuse IBGE cache, including gzip-compressed GeoJSON"""
    import geopandas as gpd

    filenames = {
        "municipio" : "malha_municipios_brasil_minima.geojson",
        "UF"        : "malha_estados_brasil_minima.geojson"   ,
    }

    if level not in filenames:
        raise ValueError("Boundary level must be 'municipio' or 'UF'.")

    path = Path(root) / ".cache/ibge" / filenames[level]

    if not path.exists():
        response = httpx.get(
            "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR",

            params ={
                "formato"      : "application/vnd.geo+json",
                "qualidade"    : "minima"                  ,
                "intrarregiao" : level                     ,
            },
            timeout=120,
        )
        response.raise_for_status()

        raw = response.content
    else:
        raw = path.read_bytes()

    payload = json.loads(
        gzip.decompress(raw)
        if   raw[:2] == b"\x1f\x8b"
        else raw
    )

    boundaries = gpd.GeoDataFrame.from_features(payload["features"], crs=4674)

    boundaries["codarea"] = boundaries["codarea"].astype("string")

    boundaries = boundaries.set_index("codarea", verify_integrity=True).to_crs(5880)

    if boundaries.empty or boundaries.index.isna().any():
        raise ValueError("Empty or unidentified IBGE geometries.")

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)

            temporary.write(raw)

        try:
            path.hardlink_to(temporary_path)
        except FileExistsError:
            pass
        finally:
            temporary_path.unlink(missing_ok=True)

    return boundaries


def _validate_coordinates(municipal):
    coordinates = municipal[["latitude", "longitude"]]

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in coordinates.dtypes):
        raise ValueError("Coordinates must be numeric.")

    if not np.isfinite(coordinates.to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite coordinates.")

    if not (
        municipal["latitude" ].between(-90 , 90 ).all() and
        municipal["longitude"].between(-180, 180).all()
    ):
        raise ValueError("Coordinates outside latitude/longitude ranges.")


def attach_centroids(municipal, boundaries):
    """Compute EPSG:5880 geometry centroids and return EPSG:4674 coordinates"""

    codes = municipal["municipality_code"]

    if not codes.is_unique or not boundaries.index.is_unique:
        raise ValueError("Repeated municipal identifier or geometry.")

    missing = codes.loc[~codes.isin(boundaries.index)]

    if not missing.empty:
        raise ValueError(f"Municipalities without IBGE geometry: {missing.tolist()}")

    geometries = boundaries.loc[codes, "geometry"]

    if geometries.isna().any() or geometries.is_empty.any() or not geometries.is_valid.all():
        raise ValueError("Missing, empty or invalid municipal geometry.")

    points = geometries.to_crs(5880).centroid.to_crs(4674)

    locations = municipal.copy()
    locations["latitude" ] = points.y.to_numpy()
    locations["longitude"] = points.x.to_numpy()

    locations = locations[MUNICIPAL_COLUMNS]

    _validate_coordinates(locations)

    return locations


def _validate_municipal(municipal, admissions):
    if list(municipal.columns) != MUNICIPAL_COLUMNS:
        raise ValueError(f"Municipal CSV columns must be exactly {MUNICIPAL_COLUMNS}.")

    if municipal.empty or municipal.isna().any().any():
        raise ValueError("Empty or incomplete municipal table.")

    if not municipal["municipality_code"].is_unique:
        raise ValueError("Repeated municipality.")

    for field in COUNT_COLUMNS:
        if not pd.api.types.is_integer_dtype(municipal[field]) or municipal[field].lt(0).any():
            raise ValueError(f"Municipal count must be a nonnegative integer: {field}.")

    _validate_coordinates(municipal)

    expected = aggregate_municipal(admissions).set_index("municipality_code").sort_index()
    actual   = municipal                      .set_index("municipality_code").sort_index()

    if set(actual.index) != set(expected.index):
        raise ValueError("Municipal universe differs from admissions.")

    for field in expected.columns:
        if not actual[field].eq(expected[field]).all():
            raise ValueError(f"Stale or inconsistent municipal table: {field}.")


def read_municipal(path, admissions):
    """Reject stale counts, changed municipality mappings and invalid locations"""

    municipal = pd.read_csv(
        path,

        dtype={
            column : "string"
            for column in [
                "municipality_code",
                "municipality"     ,
                "state_code"       ,
                "state"            ,
            ]
        }
    )

    _validate_municipal(municipal, admissions)

    return municipal.sort_values("municipality_code").reset_index(drop=True)


def observed_flows(admissions):
    """Return positive, directed observed flows, including actual local care"""

    return admissions.groupby(
        [
            "residence_code",
            "provider_code" ,
        ]
    ).size().rename(
        "observed_flow"
    ).rename_axis(
        [
            "origin_code"     ,
            "destination_code",
        ]
    ).reset_index().sort_values(
        [
            "origin_code"     ,
            "destination_code",
        ]
    ).reset_index(drop=True)


def complete_od(admissions, municipal):
    """Build the entire directed origin x observed-provider product"""

    _validate_municipal(municipal, admissions)

    origins      = municipal.loc[municipal["demand"         ].gt(0), "municipality_code"].sort_values()
    destinations = municipal.loc[municipal["observed_supply"].gt(0), "municipality_code"].sort_values()

    od = pd.MultiIndex.from_product(
        [
            origins     ,
            destinations,
        ],
        names=[
            "origin_code"     ,
            "destination_code",
        ]
    ).to_frame(index=False)

    od = od.merge(
        observed_flows(admissions),

        how     ="left"      ,
        validate="one_to_one",
    )

    od["observed_flow"] = od["observed_flow"].fillna(0).astype("int64")
    od["same_state"   ] = od["origin_code"  ].str[:2].eq(od["destination_code"].str[:2])

    return od


def validate_od(od, municipal, admissions):
    """Validate coverage, observed margins and route/null conventions together"""

    if list(od.columns) != OD_COLUMNS:
        raise ValueError(f"OD columns must be exactly {OD_COLUMNS}.")

    pairs = ["origin_code", "destination_code"]

    if od.empty or od[pairs].isna().any().any() or od.duplicated(pairs).any():
        raise ValueError("Empty, missing or repeated OD pairs.")

    if any(not od[field].map(lambda value: isinstance(value, str)).all() for field in pairs):
        raise ValueError("OD municipality codes must remain strings.")

    expected = complete_od(admissions, municipal).set_index(pairs).sort_index()

    actual   = od.assign(
        **{
            field : od[field].astype("string")
            for field in pairs
        }
    ).set_index(pairs).sort_index()

    if not actual.index.equals(expected.index):
        raise ValueError("OD coverage differs from the full origin x provider product.")

    if (
        not pd.api.types.is_integer_dtype(actual["observed_flow"])      or
            actual["observed_flow"].lt(0)                        .any() or
        not actual["observed_flow"].eq(expected["observed_flow"]).all()
    ):
        raise ValueError("Observed OD flows differ from admissions.")

    if (
        not pd.api.types.is_bool_dtype(actual["same_state"])        or
            actual["same_state"].isna()                      .any() or
        not actual["same_state"].eq  (expected["same_state"]).all()
    ):
        raise ValueError("Inconsistent same_state.")

    counts = municipal.set_index("municipality_code")

    for side, metric in [
        ("origin_code"     , "demand"         ),
        ("destination_code", "observed_supply"),
    ]:
        margin = od.groupby(side)["observed_flow"].sum().reindex(counts.index, fill_value=0)

        if not margin.eq(counts[metric]).all():
            raise ValueError(f"OD margin differs from municipal {metric}.")

    local_mask = od    ["origin_code"].eq       (od["destination_code"])
    local      = od.loc[local_mask   ].set_index("origin_code")["observed_flow"].reindex(counts.index, fill_value=0)

    if not local.eq(counts["local_care"]).all():
        raise ValueError("OD local care differs from municipal counts.")

    for side, metric in [
        ("origin_code"     , "demand"         ),
        ("destination_code", "observed_supply"),
    ]:
        crossing = od.loc[~local_mask].groupby(side)["observed_flow"].sum().reindex(counts.index, fill_value=0)

        if not crossing.eq(counts[metric] - counts["local_care"]).all():
            raise ValueError("Inconsistent outbound or inbound admissions.")

    if int(od["observed_flow"].sum()) != len(admissions):
        raise ValueError("OD flows do not sum to all admissions.")

    status = od["route_status"]

    allowed_statuses = [
        "ok"                  ,
        "intramunicipal"      ,
        "proxy_imputed"       ,
        "specific_estimate"   ,
        "municipality_missing",
        "pair_missing"        ,
        "estimate_missing"    ,
        "proxy_self_pending"  ,
        "no_route"            ,
    ]

    if status.isna().any() or not status.isin(allowed_statuses).all():
        raise ValueError("Unknown or missing route_status.")

    if not status.eq("intramunicipal").eq(local_mask).all():
        raise ValueError("Intramunicipal route_status differs from the municipality pair.")

    measures = od[["distance_km", "travel_time_min"]]

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in measures.dtypes):
        raise ValueError("Route distance and duration must be numeric.")

    absent = status.isin(["municipality_missing", "pair_missing", "no_route"])

    if not measures.loc[absent].isna().all().all():
        raise ValueError("Missing municipalities or pairs require missing distance and duration.")

    incomplete = status.isin(["estimate_missing", "proxy_self_pending"])

    if not measures.loc[incomplete].isna().any(axis=1).all():
        raise ValueError("Pending estimates require a missing distance or duration.")

    resolved = status.isin(
        [
            "ok"               ,
            "intramunicipal"   ,
            "proxy_imputed"    ,
            "specific_estimate",
        ]
    )

    if measures.loc[resolved].isna().any().any():
        raise ValueError("Resolved routes require both distance and duration.")

    values = measures.to_numpy(dtype=float, na_value=np.nan)

    if not np.isfinite(values[~np.isnan(values)]).all():
        raise ValueError("Available distances and times must be finite.")

    if measures.loc[~local_mask].le(0).any().any():
        raise ValueError("Intermunicipal distance and duration must be positive.")

    if not measures.loc[local_mask].eq(0).all().all():
        raise ValueError("Intramunicipal distance and duration must be zero.")


def flow_metrics(table, *, flow_column="observed_flow", decimals=2):
    """Admission-weighted metrics; pass decimals=None to retain full precision"""

    counts = table[flow_column].to_numpy()

    n = int(counts.sum())

    distances =       np.repeat(table["distance_km"].to_numpy(), counts )
    total     = float(np.dot   (table["distance_km"]           , counts))

    metrics = {
        "n_admissions"        : n         ,
        "n_observed_od_pairs" : len(table),
        "n_origins"           : int(table["origin_code"     ].nunique()),
        "n_destinations"      : int(table["destination_code"].nunique()),

        "total_distance_km" : total    ,
        "mean_distance_km"  : total / n,
        "max_distance_km"   : float(distances.max()),
    }

    metrics.update(
        zip(
            [
                "median_distance_km",
                "p75_distance_km"   ,
                "p90_distance_km"   ,
                "p95_distance_km"   ,
            ],

            map(
                float, np.percentile(distances, [50, 75, 90, 95], method="linear")
            ),
        )
    )

    for level in ["municipality", "state"]:
        same = int(
            table.loc[table[f"same_{level}"], flow_column].sum()
        )

        metrics.update({
            f"n_same_{level}"    : same    ,
            f"n_outside_{level}" : n - same,
        })

    return {
        key : round    (value, decimals)
        if   isinstance(value, float   ) and decimals is not None
        else value
        for key, value in metrics.items()
    }


def write_csv(table, path):
    """Atomically write UTF-8 BOM CSV without truncating floating-point precision"""

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)

    try:
        table.to_csv(
            temporary_path,

            index         =False      ,
            encoding      ="utf-8-sig",
            lineterminator="\n"       ,
        )

        temporary_path.replace(path)
    finally:
        temporary_path.unlink (missing_ok=True)
