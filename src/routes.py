"""Apply the shared, explicit corrections to the original Saldanha estimates"""

import pandas as pd

from copy import deepcopy

from src.distances import query_distances     , \
                          ROUTE_COLUMNS       , \
                          DEFAULT_PARQUET_PATH


RESOLVED_STATUSES = {
    "ok"               ,
    "intramunicipal"   ,
    "proxy_imputed"    ,
    "specific_estimate",
}
ABSENT_STATUSES   = {
    "municipality_missing",
    "pair_missing"        ,
    "no_route"            ,
}
PENDING_STATUSES  = {
    "estimate_missing"  ,
    "proxy_self_pending",
}

ROUTE_POLICY_VERSION = "saldanha_2024_proxies_v1"

PROXIES = {
    "1504752" : "1506807",  # Mojuí dos Campos -> Santarém
    "1304203" : "1300029",  # Tefé             -> Alvarães
    "1300300" : "1301100",  # Autazes          -> Careiro
}

SPECIFIC_ROUTE = {
    "origin_code"      : "1504752",
    "destination_code" : "1506807",

    "distance_km"      : 33.3,
    "travel_time_min"  : 30.0,

    "mode"             : "car"       ,
    "accessed_on"      : "2026-10-22",
    "source"           : "https://www.rome2rio.com/s/Moju%C3%AD-dos-Campos/Santar%C3%A9m",
}


def route_policy():
    """Describe the reference network and corrections for reproducible manifests"""

    return {
        "version" : ROUTE_POLICY_VERSION,

        "source"            : "https://zenodo.org/records/11400243",
        "reference_network" : "May 2024"                           ,

        "distance_unit"     : "km" ,
        "duration_unit"     : "min",
        "symmetric_source"  : True ,
        "intramunicipal_km" : 0,

        "proxy_scope"       : "unavailable original estimates; origin substitution only",
        "proxies"           : dict    (PROXIES       ),
        "specific_estimate" : deepcopy(SPECIFIC_ROUTE),
    }


def resolve_routes(raw_routes, parquet_path=DEFAULT_PARQUET_PATH):
    """
    Return corrected routes and an audit, preserving original pairs and columns.

    Only unavailable estimates may be replaced. A proxy's diagonal never gives
    an intermunicipal pair zero distance; it remains pending unless covered by
    the documented specific estimate. Already resolved routes are unchanged.
    """

    missing = set(ROUTE_COLUMNS) - set(raw_routes.columns)

    if missing:
        raise ValueError(f"Routes are missing columns: {sorted(missing)}")

    original_index = raw_routes.index
    routes         = raw_routes.copy().reset_index(drop=True)

    pending  = ~routes    ["route_status"                ].isin(RESOLVED_STATUSES)
    selected =  routes.loc[pending & routes["origin_code"].isin(PROXIES          )].copy()

    requests = selected[
        [
            "origin_code"     ,
            "destination_code",
        ]
    ].assign(
        origin_code=selected["origin_code"].map(PROXIES),
    )

    proxy_routes = (
        query_distances(
            requests    ,
            parquet_path,
        ).set_axis(selected.index)
        if   not selected.empty
        else routes.loc[selected.index, ROUTE_COLUMNS].copy()
    )

    audit = pd.concat(
        [
            selected.rename(columns={
                "route_status"    : "original_status"         ,
                "distance_km"     : "original_distance_km"    ,
                "travel_time_min" : "original_travel_time_min",
            }),

            proxy_routes.drop(
                columns="destination_code",
            ).add_prefix("proxy_"),
        ],

        axis=1,
    )

    measures = [
        "distance_km"    ,
        "travel_time_min",
    ]

    usable = proxy_routes.index[proxy_routes["route_status"].eq("ok")]

    routes.loc[usable, measures      ] = proxy_routes.loc[usable, measures]
    routes.loc[usable, "route_status"] = "proxy_imputed"

    self_proxy = proxy_routes["origin_code"].eq(proxy_routes["destination_code"])

    routes.loc[proxy_routes.index[self_proxy], "route_status"] = "proxy_self_pending"

    specific = (
        pending                                                           &
        routes["origin_code"     ].eq(SPECIFIC_ROUTE["origin_code"     ]) &
        routes["destination_code"].eq(SPECIFIC_ROUTE["destination_code"])
    )

    routes.loc[specific, measures      ] = [
        SPECIFIC_ROUTE[field]
        for field in measures
    ]
    routes.loc[specific, "route_status"] = "specific_estimate"

    audit[measures + ["route_status"]] = routes.loc[audit.index, measures + ["route_status"]]

    routes.index = original_index

    return routes, audit.reset_index(drop=True)


def prepare_routes(pairs, parquet_path=DEFAULT_PARQUET_PATH):
    """Query the original matrix and apply the policy"""

    return resolve_routes(query_distances(pairs, parquet_path), parquet_path)
