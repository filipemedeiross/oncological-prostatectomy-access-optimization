import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OBSERVED MUNICIPAL DEMAND AND SUPPLY
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Oncological prostatectomy · Brazil · January–June 2026.**

    The unit of analysis is the hospitalization (AIH and CNES), not the individual patient. **Demand** counts hospitalizations by municipality of residence without estimating clinical need; **observed supply** counts hospitalizations by municipality of care without measuring installed capacity. Local care is included in both totals. Observed flows represent actual care delivery.
    """)
    return


@app.cell
def _():
    import os
    import sys
    import json

    import numpy             as np
    import pandas            as pd
    import marimo            as mo
    import matplotlib.pyplot as plt

    from pathlib  import Path
    from hashlib  import sha256
    from datetime import datetime, timezone

    from matplotlib.patches import Patch
    from matplotlib.ticker  import PercentFormatter, ScalarFormatter

    return (
        Patch,
        Path,
        PercentFormatter,
        ScalarFormatter,
        datetime,
        json,
        mo,
        np,
        os,
        pd,
        plt,
        sha256,
        sys,
        timezone,
    )


@app.cell
def _(Path, os, sys):
    ROOT = next(
        p
        for p in (
             Path.cwd()        ,
            *Path.cwd().parents,
        )
        if (p / "notebooks").is_dir()
    )

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return (ROOT,)


@app.cell
def _():
    from src.municipal import (
        read_admissions    ,
        aggregate_municipal,
        load_boundaries    ,
        attach_centroids   ,
        observed_flows     ,
        write_csv          ,
    )

    return (
        aggregate_municipal,
        attach_centroids,
        load_boundaries,
        observed_flows,
        read_admissions,
        write_csv,
    )


@app.cell
def _(ROOT):
    START = "2026-01-01"
    END   = "2026-06-30"

    PROCEDURE = "0416010121"

    INPUT  = ROOT / "data/oncology_prostatectomy_2026_first_half.csv"
    OUTPUT = ROOT / "data/municipal_data_2026_first_half.csv"
    return END, INPUT, OUTPUT, PROCEDURE, START


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The methodology uses simplified municipal geometries from IBGE, projection to **EPSG:5880**, centroid calculation, and transformation to **EPSG:4674**. Centroids represent municipalities, not patient addresses, hospital locations, or necessarily municipal administrative centers. The cache in `.cache/ibge/` is reused; only missing files are downloaded.
    """)
    return


@app.cell
def _(
    END,
    INPUT,
    PROCEDURE,
    ROOT,
    START,
    aggregate_municipal,
    attach_centroids,
    load_boundaries,
    observed_flows,
    read_admissions,
):
    admissions = read_admissions(INPUT, START, END, PROCEDURE)

    boundaries = load_boundaries(ROOT, level="municipio")
    states     = load_boundaries(ROOT, level="UF"       )

    municipal = attach_centroids(
        aggregate_municipal(admissions),
        boundaries                     ,
    ).round(
        {
            "latitude"  : 6,
            "longitude" : 6,
        }
    )

    flows = observed_flows(admissions)
    return admissions, boundaries, flows, municipal, states


@app.cell
def _(OUTPUT, ROOT, mo, municipal, write_csv):
    write_csv(municipal, OUTPUT)

    mo.vstack(
        [
            mo.md(
                f"**{len(municipal):,} municipalities validated.**"
                f"CSV: `{OUTPUT.relative_to(ROOT)}`."
            ),

            mo.ui.table(
                municipal,

                selection=None                     ,
                label    ="Exported municipal data",
            ),
        ]
    )
    return


@app.cell
def _(admissions, municipal, np):
    analysis = municipal.set_index("municipality_code", verify_integrity=True).copy()

    analysis["outbound"  ] = analysis["demand"         ] - analysis["local_care"]
    analysis["inbound"   ] = analysis["observed_supply"] - analysis["local_care"]
    analysis["net_inflow"] = analysis["observed_supply"] - analysis["demand"    ]

    for _metric, _crossing in [
        ("demand"         , "outbound"),
        ("observed_supply", "inbound" ),
    ]:
        analysis[f"{_metric}_share_pct"] = 100 * analysis[_metric  ] / len(admissions)
        analysis[f"{_crossing}_pct"    ] = 100 * analysis[_crossing] / analysis[_metric].replace(0, np.nan)
    return (analysis,)


@app.cell
def _(admissions, analysis, flows, mo, np, pd):
    def gini(values):
        _values = np.sort(
            np.asarray(values, dtype=float)
        )

        _n = len(_values)

        return 2 * np.dot(np.arange(1, _n + 1), _values) / (_n * _values.sum()) - (_n + 1) / _n


    _volume = pd.DataFrame({
        _metric: {
            "Municipalities with positive volume" : int(analysis[_metric].gt(0).sum()),
            "Municipalities with zero volume"     : int(analysis[_metric].eq(0).sum()),

            "Admissions" : int(analysis[_metric].sum()),
        }
        for _metric in ["demand", "observed_supply"]
    })

    _concentration = pd.DataFrame({
        _metric: {
            "Gini across all municipalities"  : gini( analysis[_metric]),
            "Top 10 municipalities share (%)" : 100 * analysis[_metric].nlargest(10).sum() / len(admissions),
        }
        for _metric in ["demand", "observed_supply"]
    }).round(3)

    _positive = pd.DataFrame({
        _metric : analysis.loc[analysis[_metric].gt(0), _metric].describe()
        for _metric in ["demand", "observed_supply"]
    }).round(2)

    _interstate = flows["origin_code"].str[:2].ne(flows["destination_code"].str[:2])

    _control = pd.Series({
        "Admissions (AIHs)"      : len(admissions),
        "Municipalities (union)" : len(analysis  ),

        "Hospitals with observed production (distinct CNES)" : admissions["hospital_id"].nunique(),

        "Residence-only municipalities" : int((analysis["demand"].gt(0) & analysis["observed_supply"].eq(0)).sum()),
        "Provider-only municipalities"  : int((analysis["demand"].eq(0) & analysis["observed_supply"].gt(0)).sum()),
        "Municipalities in both roles"  : int((analysis["demand"].gt(0) & analysis["observed_supply"].gt(0)).sum()),

        "Observed OD pairs" : len(flows),

        "Local admissions"                : int(analysis["local_care"].sum()),
        "Admissions outside municipality" : int(analysis["outbound"  ].sum()),
        "Admissions outside state"        : int(flows.loc[_interstate, "observed_flow"].sum()),
    }, name="value").rename_axis("indicator").reset_index()

    mo.vstack([
        mo.md(
            "### Descriptive statistics\n\n"
            "The Gini coefficient and Lorenz curves include zeros across the common municipal universe. "
            "Descriptive statistics for positive volumes include only municipalities with admissions. "
            "The top 10 share uses the national total of admissions, without population adjustment.\n\n"

            f"Admissions outside the municipality of residence: **{analysis['outbound'].sum():,} ({100 * analysis['outbound'].sum() / len(admissions):.2f}%)**."
        ),

        mo.ui.table(
            _volume.reset_index(names="indicator"),

            selection =None ,
            pagination=False,
        ),

        mo.ui.table(
            _concentration.reset_index(names="indicator"),

            selection =None ,
            pagination=False,
        ),

        mo.ui.table(
            _positive.reset_index(names="statistic"),

            selection =None ,
            pagination=False,
            label     ="Distributions among municipalities with positive volume",
        ),

        mo.ui.table(
            _control,

            selection =None ,
            pagination=False,
            label     ="Admission volumes and flow validation",
        ),
    ])
    return


@app.cell
def _(ROOT, plt):
    COLORS = {"demand" : "#0072B2"          , "observed_supply" : "#D55E00"           }
    LABELS = {"demand" : "Demanda observada", "observed_supply" : "Produção observada"}

    LOCAL_COLOR = "#B9C4CD"

    plt.rcParams.update({
        "font.family" : "DejaVu Sans",

        "font.size"       : 9 ,
        "axes.titlesize"  : 10,
        "axes.labelsize"  : 9 ,
        "xtick.labelsize" : 8 ,
        "ytick.labelsize" : 8 ,
        "pdf.fonttype"    : 42,

        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "axes.axisbelow"    : True ,

        "axes.edgecolor"  : "#64748B",
        "axes.labelcolor" : "#172534",
        "text.color"      : "#172534",

        "savefig.facecolor" : "white",
    })


    def export_figure(fig, name, title, subtitle=""):
        fig.suptitle(
            title,

            x =0.06  ,
            y =0.99  ,
            ha="left",
            fontsize  =13    ,
            fontweight="bold",
        )

        fig.text(0.06, 0.935, subtitle, fontsize=9, color="#526170")
        fig.text(
            0.06 ,
            0.015,

            "Fonte: SIH/SUS · Prostatectomia em oncologia (0416010121) · Jan–jun de 2026.",

            fontsize=7        ,
            color   ="#526170",
        )

        fig.savefig(ROOT / "docs" / f"{name}_2026_first_half.pdf", bbox_inches="tight")

        plt.close  (fig)

        return fig

    return COLORS, LABELS, LOCAL_COLOR, export_figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SPATIAL DISTRIBUTION
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **area** of each circle is proportional to the number of admissions, using the same scale in both panels. Overlapping circles may obscure nearby municipalities.
    """)
    return


@app.cell
def _(COLORS, LABELS, analysis, boundaries, export_figure, plt, states):
    _points = boundaries.loc[analysis.index].geometry.centroid

    _fig, _axes = plt.subplots(1, 2, figsize=(8, 4.5))

    _fig.subplots_adjust(
        left  =0.04,
        right =0.98,
        bottom=0.18,
        top   =0.85,
        wspace=0.06,
    )

    for _panel, (_ax, _metric) in enumerate(zip(_axes, ["demand", "observed_supply"])):
        states.plot(ax=_ax, facecolor="#F4F6F8", edgecolor="#B0B9C1", linewidth=0.45)

        _active    = analysis.loc[analysis[_metric].gt(0)].sort_values(_metric, ascending=False)
        _locations = _points .loc[_active.index]

        _ax.scatter(
            _locations.x,
            _locations.y,

            alpha    =0.75,
            linewidth=0.25,
            edgecolor="white",

            s    =2.5 * _active[_metric],
            color=       COLORS[_metric],
        )

        _ax.set_title   (f"{chr(65 + _panel)}) {LABELS[_metric]} · {len(_active)} municípios", loc="left")
        _ax.set_axis_off()

    _handles = [
        plt.scatter(
            [],
            [],
            s    =2.5 * _n ,
            color="#64748B",
            alpha=0.75,
        )
        for _n in [1, 10, 50, 150]
    ]

    _fig.legend(
        _handles                ,
        ["1", "10", "50", "150"],

        ncol          =4             ,
        frameon       =False         ,
        loc           ="lower center",
        bbox_to_anchor=(0.5, 0.055)  ,
        title         ="Internações" ,
    )

    export_figure(
        _fig,

        "municipal_spatial_distribution"                 ,
        "Distribuição municipal da demanda e da produção",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DISTRIBUTION AND CONCENTRATION
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Panel A shows the empirical cumulative distribution of positive admission volumes, with a logarithmic scale on the admissions axis. Panel B compares Lorenz curves across the same municipal universe, including zeros and ranking each distribution independently. Concentration does not imply that the same municipalities account for the largest shares of demand and observed supply.
    """)
    return


@app.cell
def _(
    COLORS,
    LABELS,
    PercentFormatter,
    ScalarFormatter,
    analysis,
    export_figure,
    np,
    plt,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 4))

    _fig.subplots_adjust(
        left  =0.08,
        right =0.98,
        bottom=0.19,
        top   =0.82,
        wspace=0.35,
    )

    for _metric in ["demand", "observed_supply"]:
        _values   = analysis[_metric].to_numpy()
        _positive = _values [_values > 0]

        _share = np.r_[0, np.cumsum(np.sort(_values)) / _values.sum()]

        _axes[0].ecdf(_positive                             , color=COLORS[_metric], linewidth=2, label=LABELS[_metric])
        _axes[1].plot(np.linspace(0, 1, len(_share)), _share, color=COLORS[_metric], linewidth=2, label=LABELS[_metric])

    _axes[0].set(
        title="A) Municípios com volume positivo",

        xscale="log"    ,
        ylim  =(0, 1.02),

        xlabel="Internações por município (escala logarítmica)",
        ylabel="Municípios acumulados (%)"                     ,
    )
    _axes[0].set_xticks ([1, 2, 5, 10, 20, 50, 100, 200])
    _axes[0].xaxis.set_major_formatter(ScalarFormatter())

    _axes[1].plot(
        [0, 1],
        [0, 1],

        "--",

        color    ="#94A3B8",
        linewidth=1        ,
        label    ="Distribuição igualitária",
    )
    _axes[1].set(
        title=f"B) Curvas de Lorenz · {len(analysis)} municípios",

        xlabel="Municípios acumulados (%)" ,
        ylabel="Internações acumuladas (%)",

        xlim=(0, 1   ),
        ylim=(0, 1.02),
    )
    _axes[1].xaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    for _ax in _axes:
        _ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

        _ax.grid  (alpha=0.15)
        _ax.legend(loc="upper left", frameon=False, fontsize=8)

    _axes[0].legend(loc="lower right", frameon=False, fontsize=8)

    export_figure(
        _fig,

        "municipal_distribution_concentration"           ,
        "Distribuição e concentração do volume municipal",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LEADING MUNICIPALITIES AND OUT-OF-MUNICIPALITY CARE
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each panel ranks its **top 12 municipalities independently**. Gray represents local care, while the colored segment indicates admissions that cross municipal boundaries. Both panels share the same count axis, and the percentages in the titles are calculated relative to the national total of admissions.
    """)
    return


@app.cell
def _(
    COLORS,
    LABELS,
    LOCAL_COLOR,
    Patch,
    admissions,
    analysis,
    export_figure,
    plt,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    _fig.subplots_adjust(
        left  =0.18,
        right =0.98,
        bottom=0.20,
        top   =0.82,
        wspace=0.85,
    )

    for _panel, (_ax, _metric, _crossing) in enumerate(
        zip(
            _axes,

            ["demand"  , "observed_supply"],
            ["outbound", "inbound"        ],
        )
    ):
        _top = analysis.sort_values(
            [
                _metric       ,
                "municipality",
            ],
            ascending=[False, True]
        ).head(12).iloc[::-1]

        _labels = _top["municipality"] + " (" + _top["state"] + ")"

        _ax.barh(_labels, _top["local_care"], color=LOCAL_COLOR    , height=0.65)
        _ax.barh(_labels, _top[_crossing   ], color=COLORS[_metric], height=0.65, left=_top["local_care"])

        for _position, _total in enumerate(_top[_metric]):
            _ax.text(_total + 2, _position, str(_total), va="center", fontsize=8)

        _share = f"{100 * _top[_metric].sum() / len(admissions):.1f}".replace(".", ",")

        _ax.set_title(f"{chr(65 + _panel)}) {LABELS[_metric]} ({_share}% nacional)", loc="right")

        _ax.set(
            xlabel="Internações (AIHs)",
            xlim  =(0, analysis[["demand", "observed_supply"]].to_numpy().max() * 1.16),
        )

        _ax.grid       (axis="x", alpha =0.15)
        _ax.tick_params(axis="y", length=0   )

    _fig.legend(
        handles=[
            Patch(color=LOCAL_COLOR              , label="Atendimento local"             ),
            Patch(color=COLORS["demand"         ], label="Residentes atendidos fora"     ),
            Patch(color=COLORS["observed_supply"], label="Pacientes de outros municípios"),
        ],

        loc           ="lower center",
        bbox_to_anchor=(0.54, 0.03)  ,

        ncol    =3    ,
        fontsize=8    ,
        frameon =False,
    )

    export_figure(
        _fig,

        "municipal_leading_municipalities"                     ,
        "Principais municípios e atendimento fora do município",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EXECUTION METADATA
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    JSON displayed in the notebook, without an auxiliary metadata file.
    """)
    return


@app.cell
def _(
    INPUT,
    OUTPUT,
    ROOT,
    admissions,
    datetime,
    json,
    mo,
    municipal,
    sha256,
    timezone,
):
    _geometry = ROOT / ".cache/ibge/malha_municipios_brasil_minima.geojson"

    _metadata = {
        "n_admissions"     : len(admissions),
        "n_municipalities" : len(municipal ),
        "generated_at_utc" : datetime.now(timezone.utc).isoformat(timespec="seconds"),

        "source_file"     : INPUT    .relative_to(ROOT).as_posix(),
        "geometry_file"   : _geometry.relative_to(ROOT).as_posix(),
        "source_sha256"   : sha256(INPUT    .read_bytes()).hexdigest(),
        "geometry_sha256" : sha256(_geometry.read_bytes()).hexdigest(),

        "municipal_file" : OUTPUT.relative_to(ROOT).as_posix(),
    }

    mo.md(
        "```json\n" + json.dumps(_metadata, ensure_ascii=False, indent=2) + "\n```"
    )
    return


if __name__ == "__main__":
    app.run()
