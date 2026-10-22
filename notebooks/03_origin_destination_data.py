import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ORIGIN–DESTINATION DATA
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Oncological prostatectomy · Brazil · January–June 2026.**

    Define all combinations of municipalities with observed demand and municipalities with observed surgical production.

    > Source: [Saldanha, Zenodo v1](https://zenodo.org/records/11400243), **May 2024** road network. The symmetric matrix is queried using seven-digit IBGE municipality codes, regardless of origin–destination order. `dist` is converted from meters to kilometers; `dur` is already expressed in minutes.
    """)
    return


@app.cell
def _():
    import sys
    import json

    import numpy  as np
    import pandas as pd
    import marimo as mo

    from pathlib  import Path
    from hashlib  import sha256
    from tempfile import TemporaryDirectory

    return Path, TemporaryDirectory, json, mo, np, pd, sha256, sys


@app.cell
def _(Path, sys):
    ROOT = next(
        p
        for p in (
             Path.cwd()        ,
            *Path.cwd().parents,
        )
        if (p / "notebooks").is_dir()
    )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return (ROOT,)


@app.cell
def _():
    from src.distances import (
        DEFAULT_PARQUET_PATH,

        query_distances,
    )

    from src.municipal import (
        OD_COLUMNS,

        complete_od,
        validate_od,

        read_admissions,
        read_municipal ,

        write_csv,
    )

    return (
        DEFAULT_PARQUET_PATH,
        OD_COLUMNS,
        complete_od,
        query_distances,
        read_admissions,
        read_municipal,
        validate_od,
        write_csv,
    )


@app.cell
def _(ROOT):
    SOURCE    = ROOT / "data/oncology_prostatectomy_2026_first_half.csv"
    MUNICIPAL = ROOT / "data/municipal_data_2026_first_half.csv"

    OUTPUT    = ROOT / "data/origin_destination_data_2026_first_half.csv"
    INSTANCE  = ROOT / "data/telesurgery_2026_first_half.txt"
    return INSTANCE, MUNICIPAL, OUTPUT, SOURCE


@app.cell
def _(MUNICIPAL, SOURCE, complete_od, mo, read_admissions, read_municipal):
    admissions = read_admissions(SOURCE   )
    municipal  = read_municipal (MUNICIPAL, admissions)

    pairs = complete_od(admissions, municipal)

    mo.vstack(
        [
            mo.md(
                f"**{len(admissions):,} admissions and {len(municipal)} municipalities**:"
            ),
            mo.md(
                f"**{pairs['origin_code'     ].nunique()} origins × "
                f"{  pairs['destination_code'].nunique()} providers = {len(pairs):,} pairs**."
            ),

            mo.ui.table(
                municipal.head(10),

                selection=None,
                label    ="Municipalities and preserved centroids",
            ),
        ]
    )
    return admissions, municipal, pairs


@app.cell
def _(DEFAULT_PARQUET_PATH, OD_COLUMNS, pairs, query_distances):
    raw_od = pairs.merge(
        query_distances(
            pairs               ,
            DEFAULT_PARQUET_PATH,
        ),

        on=[
            "origin_code"     ,
            "destination_code",
        ],

        how     ="left"      ,
        validate="one_to_one",
    )[OD_COLUMNS]

    raw_od.head()
    return (raw_od,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PROXIES AND SPECIFIC ESTIMATE
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Only when the original estimate is unavailable do we substitute the corresponding proxy–destination pair: **Mojuí dos Campos/PA → Santarém/PA** and **Tefé/AM → Alvarães/AM**. This operation preserves the original origin and destination, flow, and `same_state`.

    The Tefé proxy is a limited approximation: the [Municipality of Alvarães](https://alvaraes.am.gov.br/2026/08/transporte-universitario-tefe-novos-horarios/) documents bus transportation to Nogueira and boat transportation between Nogueira and Tefé. Using Alvarães as a proxy does not account for the river crossing, transfers, or waiting times and does not represent patients' actual travel.

    **Mojuí dos Campos → Santarém is not assigned the proxy's zero diagonal distance.** For this specific pair, we use **33.3 km and 30 minutes by car**, according to [Rome2Rio, Mojuí dos Campos → Santarém](https://www.rome2rio.com/s/Moju%C3%AD-dos-Campos/Santar%C3%A9m), accessed on **October 22, 2026**.
    """)
    return


@app.cell
def _():
    PROXIES = {
        "1504752" : "1506807",
        "1304203" : "1300029",
    }

    SPECIFIC_ROUTE = {
        "origin_code"      : "1504752",
        "destination_code" : "1506807",

        "distance_km"     : 33.3,
        "travel_time_min" : 30.0,

        "source"      : "https://www.rome2rio.com/s/Moju%C3%AD-dos-Campos/Santar%C3%A9m",
        "accessed_on" : "2026-10-22",
        "mode"        : "car"       ,
    }
    return PROXIES, SPECIFIC_ROUTE


@app.cell
def _(
    DEFAULT_PARQUET_PATH,
    PROXIES,
    SPECIFIC_ROUTE,
    pd,
    query_distances,
    raw_od,
):
    od = raw_od.copy()

    _pending  = ~od    ["route_status"].isin(["ok", "intramunicipal"])
    _selected =  od.loc[_pending & od["origin_code"].isin(PROXIES)].copy()

    _requests = _selected[["origin_code", "destination_code"]].assign(
        origin_code=_selected["origin_code"].map(PROXIES),
    )

    _routes = query_distances(_requests, DEFAULT_PARQUET_PATH).set_axis(_selected.index)

    proxy_audit = pd.concat([
        _selected.rename(columns={
            "route_status"    : "original_status"         ,
            "distance_km"     : "original_distance_km"    ,
            "travel_time_min" : "original_travel_time_min",
        }),

        _routes.drop(columns="destination_code").add_prefix("proxy_"),
    ], axis=1)

    _usable  = _routes      ["route_status"].eq("ok")
    _indices = _routes.index[_usable]

    _measures = ["distance_km", "travel_time_min"]

    od.loc[_indices, _measures     ] = _routes.loc[_indices, _measures]
    od.loc[_indices, "route_status"] = "proxy_imputed"

    _self_proxy = _routes["origin_code"].eq(_routes["destination_code"])

    od.loc[_routes.index[_self_proxy], "route_status"] = "proxy_self_pending"

    _specific = (
        _pending &

        od["origin_code"     ].eq(SPECIFIC_ROUTE["origin_code"     ]) &
        od["destination_code"].eq(SPECIFIC_ROUTE["destination_code"])
    )

    od.loc[_specific, _measures     ] = [SPECIFIC_ROUTE[field] for field in _measures]
    od.loc[_specific, "route_status"] = "specific_estimate"

    proxy_audit[_measures + ["route_status"]] = od.loc[proxy_audit.index, _measures + ["route_status"]]

    proxy_audit = proxy_audit.reset_index(drop=True)
    return od, proxy_audit


@app.cell
def _():
    def coverage(table, stage):
        summary = table.groupby("route_status").agg(
            pairs=(
                "observed_flow",
                "size"         ,
            ),

            admissions=(
                "observed_flow",
                "sum"          ,
            ),
        ).reset_index()

        summary.insert(0, "stage", stage)

        summary["pairs_pct"] = 100 * summary["pairs"] / len(table)

        return summary

    def coverage_totals(table, stage):
        available = table[["distance_km", "travel_time_min"]].notna().all(axis=1)

        return {
            "stage" : stage     ,
            "pairs" : len(table),

            "coverage_pct"       : 100 * available.mean() ,
            "estimated_pairs"    : int(  available .sum()),
            "pending_pairs"      : int((~available).sum()),
            "pending_admissions" : int(table.loc[~available, "observed_flow"].sum()),
        }

    return (coverage_totals,)


@app.cell
def _(OUTPUT, admissions, municipal, od, validate_od, write_csv):
    validate_od(od, municipal, admissions)

    write_csv  (od, OUTPUT)

    pending_od   = od.loc[od[["distance_km", "travel_time_min"]].isna().any(axis=1)]
    validated_od = od
    return (validated_od,)


@app.cell
def _(coverage_totals, mo, od, pd, proxy_audit, raw_od):
    _audit_summary = proxy_audit.groupby(
        [
            "origin_code"      ,
            "proxy_origin_code",
            "original_status"  ,
            "route_status"     ,
        ],
    ).agg(
        pairs=(
            "observed_flow",
            "size"         ,
        ),

        admissions=(
            "observed_flow",
            "sum"          ,
        )
    ).reset_index()


    mo.vstack([
        mo.ui.table(
            pd.DataFrame([
                coverage_totals(raw_od, "before"),
                coverage_totals(od    , "after" ),
            ]),

            selection=None,
            label    ="Coverage before and after imputation",
        ),

        mo.ui.table(
            od[
                [
                    "distance_km"    ,
                    "travel_time_min",
                ]
            ].describe().reset_index().round(2),

            selection=None,
        ),
    ])
    return


@app.cell
def _(
    DEFAULT_PARQUET_PATH,
    MUNICIPAL,
    SOURCE,
    SPECIFIC_ROUTE,
    json,
    mo,
    sha256,
):
    _metadata = {
        "admissions_sha256" : sha256(SOURCE   .read_bytes()).hexdigest(),
        "municipal_sha256"  : sha256(MUNICIPAL.read_bytes()).hexdigest(),

        "source"  : "https://zenodo.org/records/11400243",
        "parquet" : str(DEFAULT_PARQUET_PATH.relative_to(DEFAULT_PARQUET_PATH.parents[3])),

        "distance_unit"    : "km" ,
        "duration_unit"    : "min",
        "symmetric_source" : True ,

        "specific_estimate" : SPECIFIC_ROUTE,
    }


    mo.md(
        "### Provenance\n\n```json\n" + json.dumps(_metadata, ensure_ascii=False, indent=2) + "\n```"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## COMPACT INSTANCE
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `ORIGINS`: IBGE municipality code and demand;

    `CANDIDATES`: IBGE municipality code and observed surgical production.

    `DISTANCES`: a matrix following these respective orders, with integer distances in kilometers, rounded using `np.rint` (round half to even), and values compatible with `uint16_t`.
    """)
    return


@app.cell
def _(municipal, np, validated_od):
    _municipal = municipal.set_index(
        "municipality_code",

        verify_integrity=True,
    ).sort_index()

    instance_origins    = _municipal.loc[_municipal["demand"         ].gt(0), "demand"         ]
    instance_candidates = _municipal.loc[_municipal["observed_supply"].gt(0), "observed_supply"]

    matrix = validated_od.pivot(
        index  ="origin_code"     ,
        columns="destination_code",
        values ="distance_km"     ,
    ).reindex(
        index =instance_origins    .index,
        columns=instance_candidates.index,
    )

    n_i, n_j = len(instance_origins), len(instance_candidates)


    _values  = matrix.to_numpy(dtype=float)

    _missing = int(
        (~np.isfinite(_values)).sum()
    )

    _rounded = np.rint(_values)

    distance_matrix = _rounded.astype(np.uint16)
    return distance_matrix, instance_candidates, instance_origins, n_i, n_j


@app.cell
def _(
    INSTANCE,
    Path,
    TemporaryDirectory,
    distance_matrix,
    instance_candidates,
    instance_origins,
    n_i,
    n_j,
    np,
):
    INSTANCE.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=INSTANCE.parent) as _directory:
        _temporary = Path(_directory) / INSTANCE.name

        with _temporary.open(
            "w",

            encoding="ascii",
            newline ="\n"   ,
        ) as _stream:
            _stream.write(f"TELESURGERY_INSTANCE 1\n{n_i} {n_j}\n\nORIGINS\n")

            instance_origins.to_csv(
                _stream,

                sep           =" "  ,
                header        =False,
                lineterminator="\n" ,
            )

            _stream.write("\nCANDIDATES\n")

            instance_candidates.to_csv(
                _stream,

                sep           =" "  ,
                header        =False,
                lineterminator="\n" ,
            )

            _stream.write("\nDISTANCES\n")

            np.savetxt(
                _stream        ,
                distance_matrix,

                fmt      ="%d",
                delimiter=" " ,
            )

        _header, _origins, _candidates, _distances = [
            block.splitlines()
            for block in _temporary.read_text(encoding="ascii").rstrip("\n").split("\n\n")
        ]

        _temporary.replace(INSTANCE)

    instance_exported = INSTANCE
    return (instance_exported,)


@app.cell(hide_code=True)
def _(
    distance_matrix,
    instance_candidates,
    instance_exported,
    instance_origins,
    mo,
    n_i,
    n_j,
):
    mo.md(f"""
    ### GENERATED INSTANCE

    `data/{instance_exported.name}`.

    **{n_i} origins × {n_j} candidates = {distance_matrix.size:,} distances**.

    Total demand: **{int(instance_origins.sum())}**;  
    Total observed production: **{int(instance_candidates.sum())}**.
    """)
    return


if __name__ == "__main__":
    app.run()
