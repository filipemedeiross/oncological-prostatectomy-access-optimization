import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OBSERVED FLOW ANALYSIS
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Oncological prostatectomy · Brazil · January–June 2026.**

    We use only origin–destination pairs with `observed_flow > 0`. The flow represents hospital admissions, not unique patients. Distances and travel times are based on previously prepared estimates, intramunicipal care retains a distance of zero. Metrics are weighted by the number of admissions.

    > The JSON file is exported to `metrics/` and the three charts as PDFs to `docs/`.
    """)
    return


@app.cell
def _():
    import sys
    import json

    import numpy             as np
    import pandas            as pd
    import marimo            as mo
    import geopandas         as gpd
    import matplotlib.pyplot as plt

    from pathlib import Path
    from hashlib import sha256
    from io      import BytesIO

    from matplotlib.patches import Patch
    from matplotlib.ticker  import MaxNLocator

    return (
        BytesIO,
        MaxNLocator,
        Patch,
        Path,
        gpd,
        json,
        mo,
        np,
        pd,
        plt,
        sha256,
        sys,
    )


@app.cell
def _(Path, sys):
    ROOT = next(
        p
        for p in (
             Path.cwd()        ,
            *Path.cwd().parents,
        )
        if (
            p / "data/origin_destination_data_2026_first_half.csv"
        ).is_file()
    )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


    from src.municipal import load_boundaries, flow_metrics

    return ROOT, flow_metrics, load_boundaries


@app.cell
def _(BytesIO, ROOT, pd):
    OD        = ROOT / "data/origin_destination_data_2026_first_half.csv"
    MUNICIPAL = ROOT / "data/municipal_data_2026_first_half.csv"
    METRICS   = ROOT / "metrics/observed_flow_metrics_2026_first_half.json"

    BARRETOS = "3505500"

    KEYS = [
        "origin_code"     ,
        "destination_code",
    ]


    od_bytes        = OD       .read_bytes()
    municipal_bytes = MUNICIPAL.read_bytes()

    _od = pd.read_csv(
        BytesIO(od_bytes),

        dtype=dict.fromkeys(KEYS, "string") | {
            "observed_flow" : "int64",
            "same_state"    : "bool" ,
        },
    )

    municipal = pd.read_csv(
        BytesIO(municipal_bytes),

        dtype={
            "municipality_code" : "string",
        },
    ).set_index("municipality_code", verify_integrity=True)


    observed = _od.loc[_od["observed_flow"].gt(0)].copy()

    observed["same_municipality"] = observed["origin_code"].eq(observed["destination_code"])

    observed.head()
    return (
        BARRETOS,
        METRICS,
        MUNICIPAL,
        OD,
        municipal,
        municipal_bytes,
        observed,
        od_bytes,
    )


@app.cell
def _(BARRETOS, flow_metrics, observed, pd):
    barretos          = observed.loc[ observed["destination_code" ].eq(BARRETOS)]
    barretos_external = barretos.loc[~barretos["same_municipality"]]

    top_barretos = barretos.loc[~barretos["same_state"]].sort_values(
        [
            "distance_km"  ,
            "observed_flow",
            "origin_code"  ,
        ],

        ascending=[False, False, True],
    ).head(10).reset_index(drop=True)


    scopes = {
        "brazil"   : observed,
        "amazonas" : observed.loc[observed["origin_code"].str.startswith("13")],
        "para"     : observed.loc[observed["origin_code"].str.startswith("15")],

        "barretos"          : barretos         ,
        "barretos_external" : barretos_external,
    }
    metrics = {
        "brazil"   : flow_metrics(observed),
        "amazonas" : flow_metrics(scopes["amazonas"]),
        "para"     : flow_metrics(scopes["para"    ]),
    }

    summary = pd.DataFrame(
        { "brazil" : flow_metrics(observed) },
    ).T.rename_axis("scope")

    summary.T.head()
    return metrics, scopes, top_barretos


@app.cell
def _(
    METRICS,
    MUNICIPAL,
    OD,
    ROOT,
    json,
    metrics,
    mo,
    municipal_bytes,
    od_bytes,
    sha256,
):
    _payload = {
        "scope_filters" : {
            "brazil" : "observed_flow > 0",
        },

        "inputs": {
            OD       .relative_to(ROOT).as_posix() : sha256(od_bytes       ).hexdigest(),
            MUNICIPAL.relative_to(ROOT).as_posix() : sha256(municipal_bytes).hexdigest(),
        },

        "metrics": metrics,
    }


    METRICS.parent.mkdir(parents=True, exist_ok=True)

    METRICS.write_text  (
        json.dumps(
            _payload,

            ensure_ascii=False,
            indent      =2    ,
            allow_nan   =False,
        ) + "\n",

        encoding="utf-8",
    )


    mo.md(f"Metrics exported to `{METRICS.relative_to(ROOT)}`.")
    return


@app.cell
def _(ROOT, gpd, load_boundaries, municipal):
    states = load_boundaries(ROOT, level="UF")

    points = gpd.GeoDataFrame(
        municipal,

        crs     =4674,
        geometry=gpd.points_from_xy(
            municipal["longitude"],
            municipal["latitude" ],
        ),
    ).to_crs(5880)
    return points, states


@app.cell
def _(ROOT, np, plt, points, states):
    COLORS = {
        "Local"                  : "#64748B",
        "Entre municípios da UF" : "#0072B2",
        "Fora da UF"             : "#D55E00",
    }


    def flow_map(ax, table, highlight, national=False):
        states.plot(
            ax=ax,

            linewidth=0.5      ,
            facecolor="#F4F6F8",
            edgecolor="#B0B9C1",
        )

        states.loc[[highlight]].plot(
            ax=ax,

            linewidth=0.8      ,
            facecolor="#E5EEF5",
            edgecolor="#64748B",
        )

        for row in table.itertuples():
            origin      = points.loc[row.origin_code     ].geometry
            destination = points.loc[row.destination_code].geometry

            color = COLORS[
                "Local"

                if row.same_municipality
                else "Entre municípios da UF"

                if row.same_state
                else "Fora da UF"
            ]

            if row.same_municipality:
                ax.scatter(
                    origin.x,
                    origin.y,

                    s         =70 ,
                    zorder    =4  ,
                    linewidths=1.8,
                    facecolors="none",
                    edgecolors=color ,
                )
            else:
                ax.plot(
                    [origin.x, destination.x],
                    [origin.y, destination.y],

                    alpha    =0.8,
                    zorder   =2  ,
                    linewidth=1.4,
                    color    =color,
                )

        codes = np.unique(
            table[["origin_code", "destination_code"]].to_numpy()
        )

        points.loc[codes].plot(
            ax=ax,

            zorder    =3 ,
            markersize=12,
            color     ="#172534",
        )

        if not national:
            west, south, east, north = states.loc[[highlight]].total_bounds
            links                    = points.loc[codes      ].total_bounds

            west  = min(west , links[0])
            south = min(south, links[1])
            east  = max(east , links[2])
            north = max(north, links[3])

            dx = (east  - west ) * 0.1
            dy = (north - south) * 0.1

            ax.set(
                xlim=(west  - dx, east  + dx),
                ylim=(south - dy, north + dy),
            )

        ax.set_axis_off()

    def flow_labels(table):
        origin       = table["origin_code"].map(points["municipality"])
        origin_state = table["origin_code"].map(points["state"       ])

        destination       = table["destination_code"].map(points["municipality"])
        destination_state = table["destination_code"].map(points["state"       ])

        return [
            f"{o} ({os}) → {d} ({ds})"
            for o, os, d, ds in zip(
                origin           ,
                origin_state     ,
                destination      ,
                destination_state,
            )
        ]

    def export_figure(fig, name):
        folder = ROOT / "docs"

        folder.mkdir(parents=True, exist_ok=True)

        fig.savefig(folder / f"{name}_2026_first_half.pdf", bbox_inches="tight")
        plt.close  (fig)

        return fig


    plt.rcParams.update({
        "font.family" : "DejaVu Sans",
        "font.size"   : 9            ,

        "axes.spines.top"   : False,
        "axes.spines.right" : False,

        "figure.titlesize"   : 20    ,
        "figure.titleweight" : "bold",

        "pdf.fonttype" : 42,
    })
    return COLORS, export_figure, flow_labels, flow_map


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## OBSERVED FLOWS · AMAZONAS, PARÁ AND BARRETOS/SP
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The points represent existing municipal centroids. Straight lines connect residence and care locations; they do not represent actual travel routes and are not used to calculate distances. Gray rings indicate local care.

    > Tefé → Manaus uses the Alvarães proxy, excluding crossings and waiting times. Mojuí dos Campos → Santarém uses the specific estimate of 33.3 km.
    """)
    return


@app.cell
def _(
    COLORS,
    MaxNLocator,
    Patch,
    export_figure,
    flow_labels,
    flow_map,
    metrics,
    np,
    plt,
    scopes,
):
    def state_figure(scope, name, uf):
        table  = scopes [scope]
        counts = metrics[scope]

        fig, (ax, bars) = plt.subplots(
            1,
            2,

            figsize=(16, max(6, 0.3 * len(table) + 2)),
            layout ="constrained"                     ,
        )

        flow_map(ax, table, uf)

        colors = np.where(
            table ["same_municipality"],
            COLORS["Local"            ],

            np.where(
                table ["same_state"            ],
                COLORS["Entre municípios da UF"],
                COLORS["Fora da UF"            ],
            )
        )

        bars.barh(
            flow_labels(table          ),
            table      ["observed_flow"],

            color=colors,
        )

        bars.invert_yaxis()

        bars.bar_label(bars.containers[0], padding=3)

        bars.set(
            xlabel="Internações (AIHs)"                   ,
            xlim  =(0, table["observed_flow"].max() * 1.2),
        )

        bars.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.legend(
            handles=[
                Patch(color=color, label=label)
                for label, color in COLORS.items()
            ],

            fontsize=8           ,
            frameon =False       ,
            loc     ="lower left",
        )

        fig.suptitle(f"Fluxos observados · {name}")

        return export_figure(fig, f"observed_flows_{scope}")

    return (state_figure,)


@app.cell
def _(state_figure):
    state_figure("amazonas", "Amazonas", "13")
    return


@app.cell
def _(state_figure):
    state_figure("para", "Pará", "15")
    return


@app.cell
def _(
    BARRETOS,
    COLORS,
    Patch,
    export_figure,
    flow_map,
    plt,
    points,
    top_barretos,
):
    _fig, (_ax, _bars) = plt.subplots(
        1,
        2,

        figsize=(16, 7)      ,
        layout ="constrained",
    )

    flow_map(_ax, top_barretos, "35", national=True)

    _target = points.loc[BARRETOS].geometry

    _ax.annotate(
        "Barretos/SP",

        (
            _target.x,
            _target.y,
        ),

        fontsize  =9        ,
        xytext    =(30, -37),
        textcoords="offset points",
        fontweight="bold"         ,
    )

    _labels = []

    for _rank, _row in enumerate(top_barretos.itertuples(), 1):
        _origin = points.loc[_row.origin_code]

        _offset = {
            3 : ( 14,  20),
            4 : (-22,  12),
            5 : ( 20,   6),
            6 : (-20, -18),
        }.get(_rank, (4, 6))

        _ax.annotate(
            str(_rank),

            (
                _origin.geometry.x,
                _origin.geometry.y,
            ),

            fontsize  =9       ,
            xytext    =_offset ,
            ha        ="center",
            textcoords="offset points",

            arrowprops={
                "arrowstyle" : "-"      ,
                "color"      : "#64748B",
                "lw"         : 0.6      ,
            },
            bbox={
                "facecolor" : "white",
                "edgecolor" : "none" ,
                "alpha"     : 0.85   ,
                "pad"       : 1      ,
            }
        )

        _labels.append(f"{_rank}. {_origin.municipality} ({_origin.state})")

    _bars.barh(_labels, top_barretos["distance_km"], color=COLORS["Fora da UF"])

    _bars.invert_yaxis()

    _bars.bar_label(
        _bars.containers[0],

        padding =4,
        fontsize=8,

        labels=[
            f"{row.distance_km:,.1f} km · {row.observed_flow} AIH(s)"
            for row in top_barretos.itertuples()
        ],
    )

    _bars.set(
        xlabel="Distância estimada até Barretos (km)"      ,
        xlim  =(0, top_barretos["distance_km"].max() * 1.4),
    )

    _bars.grid(axis="x", alpha=0.15)

    _ax.legend(
        handles=[
            Patch(
                color=COLORS["Fora da UF"],
                label="Origem fora de SP" ,
            )
        ],

        loc    ="lower left",
        frameon=False       ,
    )

    _ax  .set_title("Residência → Barretos · Linhas retas entre centroides")
    _bars.set_title(f"{len(top_barretos)} origens"                 )
    _fig .suptitle ("Deslocamentos interestaduais para Barretos/SP")

    export_figure(_fig, "observed_flows_barretos_sp")
    return


if __name__ == "__main__":
    app.run()
