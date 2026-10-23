"""
Nearest-provider allocation and its impact, using the prepared OD measures.

Counts describe admissions and observed production, never provider capacity.
"""

from pathlib import Path
from hashlib import sha256
from io      import BytesIO

import numpy  as np
import pandas as pd

from src.municipal import COUNT_COLUMNS, flow_metrics


KEYS = [
    "origin_code"     ,
    "destination_code",
]

ATOL = 1e-6

RESOLVED_STATUSES = {"ok"                  , "intramunicipal"    , "proxy_imputed", "specific_estimate"}
ABSENT_STATUSES   = {"municipality_missing", "pair_missing"      , "no_route"     }
PENDING_STATUSES  = {"estimate_missing"    , "proxy_self_pending"}


def _integer_counts(values, name, *, positive=False):
    if (
        not pd.api.types.is_integer_dtype(values)             or
            values.isna()                              .any() or
           (values.le(0) if positive else values.lt(0)).any()
    ):
        qualifier = "positive" if positive else "nonnegative"

        raise ValueError(f"{name} must contain {qualifier} integers.")


def _validate_routes(od):
    if (
        od.empty                    or
        od[KEYS].isna().any().any() or
        od.duplicated(KEYS)  .any()
    ):
        raise ValueError("Empty, missing or repeated OD pairs.")

    for key in KEYS:
        if not od[key].astype("string").str.fullmatch(r"[0-9]{7}" ).all():
            raise ValueError(f"Invalid IBGE identifier: {key}.")

        if not od[key].map   (lambda value: isinstance(value, str)).all():
            raise ValueError("OD municipality codes must remain strings.")

    _integer_counts(od["observed_flow"], "observed_flow")

    local          = od["origin_code"]        .eq(od["destination_code"])
    expected_state = od["origin_code"].str[:2].eq(od["destination_code"].str[:2])

    if (
        not pd.api.types.is_bool_dtype(od["same_state"]) or
            od["same_state"].isna()               .any() or
        not od["same_state"].eq  (expected_state) .all()
    ):
        raise ValueError("Inconsistent same_state flag.")

    status = od["route_status"]

    if status.isna().any() or not status.isin(
        RESOLVED_STATUSES |
        ABSENT_STATUSES   |
        PENDING_STATUSES
    ).all():
        raise ValueError("Unknown or missing route_status.")

    if not status.eq("intramunicipal").eq(local).all():
        raise ValueError("Intramunicipal status differs from the municipality pair.")

    measures = od[["distance_km", "travel_time_min"]]

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in measures.dtypes
    ):
        raise ValueError("Distance and travel time must be numeric.")

    values = measures.to_numpy(dtype=float, na_value=np.nan)

    if not np.isfinite(values[~np.isnan(values)]).all():
        raise ValueError("Available route measures must be finite.")

    if not measures.loc[status.isin(ABSENT_STATUSES)].isna().all().all():
        raise ValueError("Unavailable routes require missing distance and travel time.")

    if not measures.loc[status.isin(PENDING_STATUSES)].isna().any(axis=1).all():
        raise ValueError("Pending estimates require a missing route measure.")

    if measures.loc[status.isin(RESOLVED_STATUSES)].isna().any().any():
        raise ValueError("Resolved routes require distance and travel time.")

    if measures.loc[~local].le(0).any().any():
        raise ValueError("Intermunicipal route measures must be positive.")

    if not measures.loc[local].eq(0).all().all():
        raise ValueError("Intramunicipal route measures must be zero.")


def _available_routes(od):
    return (
        od["route_status"].isin(RESOLVED_STATUSES) &
        od[
            [
                "distance_km"    ,
                "travel_time_min",
            ]
        ].notna().all(axis=1)
    )


def route_audit(od):
    """Count route availability without silently dropping observed admissions"""

    unavailable = ~_available_routes(od)

    return {
        "n_od_pairs" : len(od),

        "n_available_pairs"   : int((~unavailable).sum()),
        "n_unavailable_pairs" : int(  unavailable .sum()),

        "route_status_counts": {
            str(status) : int(count)
            for status, count in od["route_status"].value_counts().sort_index().items()
        },
    }


def load_inputs(od_path, municipal_path):
    """Read current CSVs and reconcile their universes, margins and local care"""

    sources      = {
        "od"        : Path(od_path       ),
        "municipal" : Path(municipal_path),
    }
    source_bytes = {
        name : path.read_bytes()
        for name, path in sources.items()
    }

    od = pd.read_csv(
        BytesIO(source_bytes["od"]),

        dtype=dict.fromkeys(KEYS, "string") | {
            "observed_flow" : "int64"  ,
            "same_state"    : "boolean",
            "route_status"  : "string" ,
        },
    )
    municipal = pd.read_csv(
        BytesIO(source_bytes["municipal"]),

        dtype=dict.fromkeys(
            [
                "municipality_code",
                "municipality"     ,
                "state_code"       ,
                "state"            ,
            ],
            "string"
        ) | dict.fromkeys(COUNT_COLUMNS, "int64"),
    )

    _validate_routes(od)

    if municipal.empty or municipal.isna().any().any():
        raise ValueError("Empty or incomplete municipal table.")

    municipal = municipal.set_index("municipality_code", verify_integrity=True).sort_index()

    origins   = municipal.index[municipal["demand"         ].gt(0)]
    providers = municipal.index[municipal["observed_supply"].gt(0)]

    if (
        not len(origins) or not len(providers) or

        set(od["origin_code"     ]) != set(origins  ) or
        set(od["destination_code"]) != set(providers) or

        len(od) != len(origins) * len(providers)
    ):
        raise ValueError("OD coverage differs from the full origin x observed-provider product.")

    for key, count in zip(KEYS, ["demand", "observed_supply"]):
        margin = od.groupby(key)["observed_flow"].sum().reindex(municipal.index, fill_value=0)

        if not margin.eq(municipal[count]).all():
            raise ValueError(f"Observed OD margin differs from municipal {count}.")

    local        = od.loc[od["origin_code"].eq(od["destination_code"])]

    local_counts = local.set_index("origin_code")["observed_flow"].reindex(municipal.index, fill_value=0)

    if not local_counts.eq(municipal["local_care"]).all():
        raise ValueError("Observed local care differs from municipal counts.")

    if route_audit(od)["n_observed_pairs_missing_measures"]:
        raise ValueError("Observed admissions have missing distance or travel time.")

    municipal["label"] = municipal["municipality"] + "/" + municipal["state"]

    od = od.sort_values(KEYS).reset_index(drop=True)

    od.attrs["route_audit"] = route_audit(od)
    od.attrs["sources"    ] = {
        name : {
            "path"   : str(path)                             ,
            "sha256" : sha256(source_bytes[name]).hexdigest(),
        }
        for name, path in sources.items()
    }

    return od, municipal


def observed_baseline(od):
    """Retain every positive observed flow and require complete route measures"""

    _validate_routes(od)

    observed = od.loc[od["observed_flow"].gt(0)].copy()

    observed["same_municipality"] = observed["origin_code"  ].eq(observed["destination_code"])
    observed["total_distance_km"] = observed["observed_flow"] *  observed["distance_km"     ]

    observed = observed.sort_values(KEYS).reset_index(drop=True)

    observed.attrs.update(
        n_available_providers=int(od["destination_code"].nunique()),
        route_audit          =route_audit(od)                      ,
    )

    return observed


def nearest_provider(od, demand_series):
    """Allocate all demand to the shortest measured route; break ties by IBGE"""

    _validate_routes(od)
    _integer_counts (demand_series, "demand")

    demand = demand_series.loc[demand_series.gt(0)].copy()

    if demand.empty or not demand.index.is_unique or demand.index.isna().any():
        raise ValueError("Demand must have unique origin identifiers and a positive total.")

    if not demand.index.map(lambda value: isinstance(value, str)).all():
        raise ValueError("Demand origin codes must remain strings.")

    available = od.loc[
        od["origin_code"].isin(demand.index) & _available_routes(od)
    ]

    nearest = available.sort_values(
        [
            "origin_code"     ,
            "distance_km"     ,
            "destination_code",
        ]
    ).drop_duplicates("origin_code")

    if set(nearest["origin_code"]) != set(demand.index):
        missing = sorted(set(demand.index) - set(nearest["origin_code"]))

        raise ValueError(f"Origins without a measured provider alternative: {missing}.")

    scenario = nearest.drop(columns="observed_flow").copy()

    scenario["allocated_flow"   ] = scenario["origin_code"   ].map(demand).astype("int64"      )
    scenario["same_municipality"] = scenario["origin_code"   ].eq (scenario["destination_code"])
    scenario["total_distance_km"] = scenario["allocated_flow"] * scenario["distance_km"]

    scenario = scenario[KEYS + [
        "allocated_flow"   ,
        "distance_km"      ,
        "travel_time_min"  ,
        "route_status"     ,
        "total_distance_km",
        "same_municipality",
        "same_state"       ,
    ]].sort_values(KEYS).reset_index(drop=True)

    if (
        not scenario.set_index("origin_code")["allocated_flow"].reindex(demand.index).eq(demand).all() or
        not np.isfinite(scenario["total_distance_km"]).all()
    ):
        raise ValueError("Allocation failed to preserve demand or finite weighted distances.")

    scenario.attrs.update(
        n_available_providers=int(od["destination_code"].nunique()),
        route_audit          =route_audit(od)                      ,
    )

    return scenario


def scenario_metrics(table, flow_column):
    """Admission-weighted distance metrics, retaining full floating precision"""

    _integer_counts(table[flow_column], flow_column)

    active = table.loc[table[flow_column].gt(0)]

    if active.empty or not np.isfinite(active["distance_km"]).all() or active["distance_km"].lt(0).any():
        raise ValueError("Metrics require positive admissions with finite nonnegative distances.")

    for level in ["municipality", "state"]:
        flag = active[f"same_{level}"]

        if not pd.api.types.is_bool_dtype(flag) or flag.isna().any():
            raise ValueError(f"Metrics require complete boolean same_{level} flags.")

    metrics = flow_metrics(active, flow_column=flow_column, decimals=None)

    metrics["n_active_od_pairs"] = metrics.pop("n_observed_od_pairs")
    metrics["n_used_providers" ] = metrics.pop("n_destinations"     )
    metrics["n_providers"      ] = int(
        table.attrs.get("n_available_providers", metrics["n_used_providers"])
    )

    n = metrics["n_admissions"]

    for level in ["municipality", "state"]:
        metrics.update({
            f"pct_same_{level}"    : 100 * metrics[f"n_same_{level}"   ] / n,
            f"pct_outside_{level}" : 100 * metrics[f"n_outside_{level}"] / n,
        })

    return metrics


def compare_metrics(observed_metrics, nearest_metrics):
    """Reconcile admission counts and report reductions against the baseline"""

    if observed_metrics["n_admissions"] != nearest_metrics["n_admissions"]:
        raise ValueError("Observed and reallocated admission totals differ.")

    total     =         observed_metrics["total_distance_km"]
    mean      =         observed_metrics["mean_distance_km" ]
    reduction = total - nearest_metrics ["total_distance_km"]

    if reduction < -ATOL:
        raise ValueError("Nearest-provider allocation is worse than the observed baseline.")

    return {
        "baseline_total_distance_km"     : total    ,
        "baseline_mean_distance_km"      : mean     ,
        "absolute_distance_reduction_km" : reduction,
        "mean_distance_reduction_km"     : mean - nearest_metrics["mean_distance_km"],

        "distance_reduction_pct"      : 100 * reduction / total                                   if total else 0.0,
        "mean_distance_reduction_pct" : 100 * (mean - nearest_metrics["mean_distance_km"]) / mean if mean  else 0.0,
    }


def impact_tables(observed, scenario, municipal, top_n=10):
    """Decompose national savings into complete flow/provider tables and rankings"""

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    if scenario["origin_code"].duplicated().any():
        raise ValueError("Impact analysis requires one scenario provider per origin.")

    expected  = observed.groupby  ("origin_code")["observed_flow" ].sum().sort_index()
    allocated = scenario.set_index("origin_code")["allocated_flow"]      .sort_index()

    if not expected.index.equals(allocated.index) or not expected.eq(allocated).all():
        raise ValueError("Impact analysis must compare the same admissions at every origin.")

    if not municipal.index.is_unique:
        raise ValueError("Repeated municipality in impact reference.")

    flows = observed[
        KEYS + [
            "observed_flow"  ,
            "distance_km"    ,
            "travel_time_min",
            "route_status"   ,
        ]
    ].rename(columns={
        "destination_code" : "observed_destination_code",
        "distance_km"      : "observed_distance_km"     ,
        "travel_time_min"  : "observed_travel_time_min" ,
        "route_status"     : "observed_route_status"    ,
    }).merge(
        scenario[
            KEYS + [
                "distance_km"    ,
                "travel_time_min",
                "route_status"   ,
            ]
        ].rename(columns={
            "destination_code" : "nearest_destination_code",
            "distance_km"      : "nearest_distance_km"     ,
            "travel_time_min"  : "nearest_travel_time_min" ,
            "route_status"     : "nearest_route_status"    ,
        }),

        on      ="origin_code",
        how     ="left"       ,
        validate="many_to_one",
    )

    if not np.isfinite(flows[["observed_distance_km", "nearest_distance_km"]]).all().all():
        raise ValueError("Impact analysis requires measured observed and nearest distances.")

    flows["destination_changed"  ] = flows["observed_destination_code"].ne(flows["nearest_destination_code"])
    flows["distance_reduction_km"] = flows["observed_distance_km"     ] -  flows["nearest_distance_km"     ]

    if flows["distance_reduction_km"].lt(-ATOL).any():
        raise ValueError("An allocated provider is farther than the observed provider.")

    flows["saved_admission_km"    ] = flows["observed_flow"] * flows["distance_reduction_km"]
    flows["distance_reduction_pct"] = (
        100 * flows["distance_reduction_km"] / flows["observed_distance_km"].replace(0, np.nan)
    ).fillna(0.0)

    for prefix, column in [
        ("origin"  , "origin_code"              ),
        ("observed", "observed_destination_code"),
        ("nearest" , "nearest_destination_code" ),
    ]:
        flows[f"{prefix}_label"] = flows[column].map(municipal["label"])
        flows[f"{prefix}_state"] = flows[column].map(municipal["state"])

        if flows[[f"{prefix}_label", f"{prefix}_state"]].isna().any().any():
            raise ValueError("Impact flow contains an unknown municipality.")

    provider_codes = municipal.index[municipal["observed_supply"].gt(0)]

    if not scenario["destination_code"].isin(provider_codes).all():
        raise ValueError("Scenario allocates admissions outside the observed provider set.")

    providers = municipal.loc[provider_codes, ["municipality", "state", "label"]].copy()

    providers["observed_admissions"] = observed.groupby(
        "destination_code"
    )["observed_flow"].sum().reindex(provider_codes, fill_value=0)

    if not providers["observed_admissions"].eq(municipal.loc[provider_codes, "observed_supply"]).all():
        raise ValueError("Observed provider counts differ from the municipal reference.")

    providers["nearest_admissions"] = scenario.groupby(
        "destination_code"
    )["allocated_flow"].sum().reindex(provider_codes, fill_value=0)

    changed = flows.loc[flows["destination_changed"]]

    for column, code in [
        ("incoming_admissions", "nearest_destination_code" ),
        ("outgoing_admissions", "observed_destination_code"),
    ]:
        providers[column] = changed.groupby(code)["observed_flow"].sum().reindex(provider_codes, fill_value=0)

    providers["retained_admissions"] = flows.loc[
        ~flows["destination_changed"]
    ].groupby(
        "observed_destination_code"
    )["observed_flow"].sum().reindex(provider_codes, fill_value=0)

    providers["net_increase"] = providers["nearest_admissions"] - providers["observed_admissions"]
    providers["increase_pct"] = 100 * providers["net_increase"] / providers["observed_admissions"]

    providers["saved_admission_km"] = flows.groupby(
        "nearest_destination_code"
    )["saved_admission_km"].sum().reindex(provider_codes, fill_value=0)

    saved_total = float(flows["saved_admission_km"].sum())

    providers["saving_share_pct"] = 100 * providers["saved_admission_km"] / saved_total if saved_total else 0.0

    if (
        not providers["observed_admissions"].eq(providers["retained_admissions"] + providers["outgoing_admissions"]).all() or
        not providers["nearest_admissions" ].eq(providers["retained_admissions"] + providers["incoming_admissions"]).all() or

        int(providers["net_increase"].sum()) != 0
    ):
        raise ValueError("Provider incoming, outgoing and retained admissions do not reconcile.")

    observed_total = float((observed["observed_flow" ] * observed["distance_km"]).sum())
    nearest_total  = float((scenario["allocated_flow"] * scenario["distance_km"]).sum())

    if not np.allclose(
        [
            saved_total                          ,
            providers["saved_admission_km"].sum(),
        ],

        observed_total - nearest_total,

        rtol=1e-12,
        atol=ATOL ,
    ):
        raise ValueError("Flow and provider savings do not reconcile with national savings.")

    providers = providers.rename_axis("municipality_code").reset_index()

    flow_ranking = flows.loc[
        flows["destination_changed"] & flows["distance_reduction_km"].gt(ATOL)
    ].sort_values(
        [
            "distance_reduction_km"    ,
            "saved_admission_km"       ,
            "origin_code"              ,
            "observed_destination_code",
        ],

        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    flow_ranking.insert(0, "distance_rank", np.arange(1, len(flow_ranking) + 1))

    growth = providers.loc[providers["net_increase"].gt(0)]

    return {
        "flows"        : flows       ,
        "providers"    : providers   ,
        "flow_ranking" : flow_ranking,

        "top_growth"     : growth.sort_values(["net_increase",                 "municipality_code"], ascending=[False, True       ]).head(top_n),
        "top_growth_pct" : growth.sort_values(["increase_pct", "net_increase", "municipality_code"], ascending=[False, False, True]).head(top_n),

        "top_saving" : providers.loc[
            providers["saved_admission_km"].gt(ATOL)
        ].sort_values(
            [
                "saved_admission_km",
                "municipality_code" ,
            ],

            ascending=[False, True]
        ).head(top_n),

        "top_flows"          : flow_ranking.head       (top_n),
        "top_weighted_flows" : flow_ranking.sort_values(
            [
                "saved_admission_km"       ,
                "distance_reduction_km"    ,
                "origin_code"              ,
                "observed_destination_code",
            ],

            ascending=[False, False, True, True],
        ).head(top_n),

        "audit" : {
            "n_admissions"             : int(observed["observed_flow"].sum()),
            "n_changed_admissions"     : int(changed ["observed_flow"].sum()),
            "n_changed_observed_flows" : len(changed),

            "observed_admission_km" : observed_total,
            "nearest_admission_km"  : nearest_total ,
            "saved_admission_km"    : saved_total   ,
        },
    }
