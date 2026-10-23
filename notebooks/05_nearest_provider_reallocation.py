import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # UNCONSTRAINED NEAREST-PROVIDER REALLOCATION
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Scenario 1 · Oncological prostatectomy · Brazil · January–June 2026.**

    Assign each municipality's observed demand to its nearest observed provider municipality, without capacity constraints. This minimizes admission-weighted distance to a fixed provider set; observed production does not represent available capacity. The observed baseline and both reallocation scenarios use the same route estimates.
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

    from textwrap        import fill
    from pathlib         import Path
    from hashlib         import sha256
    from collections.abc import Mapping

    from matplotlib.lines       import Line2D
    from matplotlib.figure      import Figure
    from matplotlib.collections import LineCollection
    from matplotlib.patches     import FancyArrowPatch
    from matplotlib.ticker      import FuncFormatter  , MaxNLocator

    return (
        FancyArrowPatch,
        Figure,
        FuncFormatter,
        Line2D,
        LineCollection,
        Mapping,
        MaxNLocator,
        Path,
        fill,
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
    return (ROOT,)


@app.cell
def _():
    from src.municipal    import STATE_CODES, load_boundaries, write_csv

    from src.reallocation import (
        load_inputs      ,
        observed_baseline,
        route_audit      ,
        compare_metrics  ,
        impact_tables    ,
        nearest_provider ,
        scenario_metrics ,
    )

    return (
        STATE_CODES,
        compare_metrics,
        impact_tables,
        load_boundaries,
        load_inputs,
        nearest_provider,
        observed_baseline,
        route_audit,
        scenario_metrics,
        write_csv,
    )


@app.cell
def _(ROOT):
    PERIOD      = "2026_first_half"

    TOP_N       = 10

    CASE_STATES = ("MA", "PA")

    OD        = ROOT / f"data/origin_destination_data_{PERIOD}.csv"
    MUNICIPAL = ROOT / f"data/municipal_data_{PERIOD}.csv"

    RESULT  = ROOT / f"results/scenario_1_{PERIOD}.csv"
    METRICS = ROOT / f"metrics/scenario_1_metrics_{PERIOD}.json"
    FIGURES = ROOT / "docs/"
    return CASE_STATES, FIGURES, METRICS, MUNICIPAL, OD, PERIOD, RESULT, TOP_N


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## INPUTS AND NATIONAL ALLOCATION
    """)
    return


@app.cell
def _(MUNICIPAL, OD, load_inputs, observed_baseline, route_audit):
    od, municipal = load_inputs      (OD, MUNICIPAL)
    observed      = observed_baseline(od)
    coverage      = route_audit      (od)

    input_sources = od.attrs["sources"]

    coverage
    return input_sources, municipal, observed, od


@app.cell
def _(municipal, nearest_provider, od):
    demand = municipal.loc[municipal["demand"].gt(0), "demand"]

    scenario = nearest_provider(od, demand)

    scenario.head(10)
    return (scenario,)


@app.cell
def _(compare_metrics, observed, scenario, scenario_metrics):
    observed_metrics = scenario_metrics(observed, "observed_flow" )
    metrics          = scenario_metrics(scenario, "allocated_flow")

    reductions = compare_metrics(observed_metrics, metrics)

    reductions
    return metrics, observed_metrics, reductions


@app.cell
def _(FIGURES, PERIOD, plt):
    def export_figure(figure, name):
        """Save the displayed figure without mutating shared notebook state"""

        if figure is None:
            return None, []

        FIGURES.mkdir(parents=True, exist_ok=True)

        paths = [
            FIGURES / f"{name}_{PERIOD}.{suffix}"
            for suffix in ("pdf",)
        ]

        for path in paths:
            options = {
                "metadata" : {
                    "CreationDate" : None,
                    "ModDate"      : None,
                }
            } if path.suffix == ".pdf" else {}

            figure.savefig(path, dpi=160, bbox_inches="tight", **options)

        plt.close(figure)

        return figure, paths

    return (export_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NATIONAL COMPARISON
    """)
    return


@app.cell
def _(Figure, plt):
    COLORS = {
        "observed" : "#BE672D",
        "nearest"  : "#0072B2",
        "saving"   : "#168475",
        "text"     : "#243746",
    }

    PLOT_STYLE = {
        "font.family"       : "DejaVu Sans",
        "font.size"         : 10,
        "pdf.fonttype"      : 42,

        "text.color"        : COLORS["text"],
        "axes.labelcolor"   : COLORS["text"],

        "figure.facecolor"  : "white",
        "axes.titleweight"  : "bold" ,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
    }

    def br(value: float, decimals: int = 0) -> str:
        """Format a numeric value with Brazilian decimal and thousands separators"""

        return f"{value:,.{decimals}f}".translate(str.maketrans({",": ".", ".": ","}))

    def empty_figure(figsize, title, message, subtitle="", footer="") -> Figure:
        """Keep an empty selection visible with the same typography and margins"""

        fig, ax = plt.subplots(figsize=figsize)

        fig.subplots_adjust(
            left  =0.04,
            right =0.97,
            top   =0.82,
            bottom=0.12,
        )

        ax.set_axis_off()

        ax.text(
            0.5,
            0.5,

            message,

            fontsize =12,
            ha       ="center" ,
            va       ="center" ,
            color    ="#52616D",
            transform=ax.transAxes,
        )

        fig.suptitle(title, x=0.04, y=0.97, fontsize=17, ha="left", weight="bold")

        if subtitle:
            fig.text(0.04, 0.915, subtitle, fontsize=11)
        if footer:
            fig.text(0.04, 0.025, footer  , fontsize=9 , color="#52616D")

        return fig

    return COLORS, PLOT_STYLE, br, empty_figure


@app.cell
def _(Figure, Mapping, PLOT_STYLE, np, plt):
    def national_comparison(
        observed_metrics: Mapping[str, float],
        scenario_metrics: Mapping[str, float],
    ) -> Figure:
        """Compare admission-weighted national measures and allocation percentages"""

        with plt.rc_context(PLOT_STYLE):
            baseline_total = observed_metrics["total_distance_km"]
            total_distance = scenario_metrics["total_distance_km"]
            baseline_mean  = observed_metrics["mean_distance_km" ]
            mean_distance  = scenario_metrics["mean_distance_km" ]

            reduction_pct  = (
                100 * (baseline_total - total_distance) / baseline_total
                if baseline_total > 0 else 0.0
            )

            interstate_change_pp = (
                scenario_metrics["pct_outside_state"] - observed_metrics["pct_outside_state"]
            )

            comparison_colors = (
                "#7A8793",
                "#0072B2",
            )

            fig = plt.figure(figsize=(14, 7.4), facecolor="white")

            layout = fig.add_gridspec(
                2,
                6,

                hspace       =0.7,
                wspace       =1.4,
                height_ratios=[1, 2.8],
            )

            fig.subplots_adjust(
                left  =0.075,
                right =0.97 ,
                top   =0.85 ,
                bottom=0.15 ,
            )

            fig.suptitle(
                "Comparação · Observado → Cenário 1",

                x       =0.075 ,
                y       =0.98  ,
                fontsize=16    ,
                ha      ="left",
            )

            fig.text(
                0.075,
                0.92 ,

                f"{scenario_metrics['n_admissions']:,} admissões",

                fontsize=10       ,
                color   ="#52616D",
            )

            cards = (
                (
                    0,

                    "DISTÂNCIA TOTAL"                                  ,
                    f"{reduction_pct:.2f}%"                            ,
                    f"{baseline_total:,.1f} → {total_distance:,.1f} km",

                    comparison_colors[1],
                ),

                (
                    2,

                    "DISTÂNCIA MÉDIA POR ADMISSÃO"                 ,
                    f"{baseline_mean - mean_distance:.2f}"         ,
                    f"{baseline_mean:.2f} → {mean_distance:.2f} km",

                    comparison_colors[1],
                ),

                (
                    4,

                    "ADMISSÕES FORA DO ESTADO"    ,
                    f"{interstate_change_pp:+.2f}",
                    f"{observed_metrics['n_outside_state']:,} → "
                    f"{scenario_metrics['n_outside_state']:,} admissões "
                    f"({observed_metrics['pct_outside_state']:.2f}% → "
                    f"{scenario_metrics['pct_outside_state']:.2f}%)"    ,

                    "#8A5318",
                ),
            )

            for position, label, headline, detail, color in cards:
                ax = fig.add_subplot(layout[0, position:position + 2])

                ax.set_facecolor("#F3F6F8")

                ax.text(0.04, 0.8 , label   , fontsize=9 , color="#52616D", transform=ax.transAxes)
                ax.text(0.04, 0.43, headline, fontsize=20, color=color    , transform=ax.transAxes, fontweight="bold")
                ax.text(0.04, 0.12, detail  , fontsize=10,                  transform=ax.transAxes)

                ax.set(xticks=[], yticks=[])

                ax.spines[:].set_visible(False)

            panels = (
                (
                    slice(0, 3),

                    "A) Distância por admissão",

                    ["mean_distance_km", "median_distance_km", "p75_distance_km", "p90_distance_km", "p95_distance_km"],
                    ["Média"           , "Mediana"           , "P75"            , "P90"            , "P95"            ],

                    "km",
                ),

                (
                    slice(3, 6),

                    "B) Alocação de admissões",

                    ["pct_same_municipality", "pct_outside_state"],
                    ["No próprio município" , "Fora do estado"   ],

                    "%",
                ),
            )

            for columns, title, fields, labels, unit in panels:
                ax = fig.add_subplot(layout[1, columns])

                positions = np.arange(len(fields))

                for offset, values, color, label in (
                    (-0.18, observed_metrics, comparison_colors[0], "Fluxo observado"),
                    ( 0.18, scenario_metrics, comparison_colors[1], "Cenário 1"      ),
                ):
                    bars = ax.barh(
                        positions + offset,

                        [
                            values[field]
                            for field in fields
                        ],

                        zorder=3    ,
                        height=0.32 ,
                        color =color,
                        label =label,
                    )

                    ax.bar_label(
                        bars,

                        labels=[
                            f"{values[field]:.2f}{'%' if unit == '%' else ''}"
                            for field in fields
                        ],

                        padding =5,
                        fontsize=9,
                    )

                limit = max(
                    max(observed_metrics[field] for field in fields),
                    max(scenario_metrics[field] for field in fields),
                )

                ax.set(
                    yticks     =positions,
                    yticklabels=labels   ,

                    xlim  =(0, max(limit * 1.25, 1)),
                    xlabel=unit                     ,
                )

                ax.invert_yaxis()

                ax.set_title(title, loc="left", fontsize=12, pad=14)

                ax.spines[:].set_visible(False)

                ax.tick_params(axis="both", length=0, labelsize=10)

                ax.xaxis.grid(True, color="#E5EAF0", linewidth=0.7)

            fig.legend(
                *ax.get_legend_handles_labels(),

                ncol          =2 ,
                fontsize      =10,
                frameon       =False,
                loc           ="lower center",
                bbox_to_anchor=(0.5, 0.045)  ,
            )

            return fig

    return (national_comparison,)


@app.cell
def _(export_figure, metrics, national_comparison, observed_metrics):
    _figure, comparison_paths = export_figure(
        national_comparison(
            observed_metrics,
            metrics         ,
        ),

        "national_comparison",
    )

    _figure
    return (comparison_paths,)


@app.cell
def _(metrics, mo, observed_metrics, pd):
    mo.ui.table(
        pd.DataFrame(
            {
                "observed"         : observed_metrics,
                "nearest_provider" : metrics         ,
            }
        )
        .rename_axis("metric")
        .reset_index(),

        selection=None,
        label    ="National metrics (admission-weighted distances in km)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IMPACT BY PROVIDER MUNICIPALITY AND OBSERVED FLOW
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For each observed origin–provider flow, compare its distance with the national nearest-provider distance for that origin:

    $$\Delta d_{ij}=d^{obs}_{ij}-d_{i,j^*(i)},\qquad E_{ij}=f_{ij}\Delta d_{ij}.$$

    Savings are attributed to the new provider municipality. This additive decomposition reconciles with the national reduction and does not estimate the causal effect of opening or removing a provider. Rankings show the top 10; complete provider and flow tables remain available below.
    """)
    return


@app.cell
def _(TOP_N, impact_tables, municipal, observed, scenario):
    impact = impact_tables(observed, scenario, municipal, top_n=TOP_N)
    return (impact,)


@app.cell
def _(
    COLORS,
    Figure,
    FuncFormatter,
    MaxNLocator,
    PLOT_STYLE,
    br,
    empty_figure,
    np,
    pd,
    plt,
):
    def provider_growth(frame: pd.DataFrame, percentage: bool = False) -> Figure:
        """Draw a provider growth ranking in its supplied order, with volume details"""

        with plt.rc_context(PLOT_STYLE):
            title = (
                f"{len(frame)} municípios provedores com maior aumento percentual"
                if   percentage
                else f"{len(frame)} municípios provedores com maior acréscimo"
            )

            subtitle = (
                "Variação relativa à oferta observada"
                if   percentage
                else "Cenário sem restrições de capacidade"
            )

            footer = (
                "SIH/SUS · Jan-jun/2026 · Cenário sem restrições de capacidade."
                if   percentage
                else "SIH/SUS · Jan-jun/2026 · Aumento não implica capacidade disponível."
            )

            if frame.empty:
                return empty_figure(
                    (13.5, 8.8),
                    title      ,

                    "Nenhum provedor apresentou acréscimo de internações.",

                    subtitle,
                    footer  ,
                )

            fig, (ax, detail) = plt.subplots(
                1, 2, figsize=(13.5, 8.8), gridspec_kw={"width_ratios": [3.5, 1.9]},
            )

            fig.subplots_adjust(
                left  =0.25,
                right =0.97,
                top   =0.84,
                bottom=0.10,
                wspace=0.17,
            )

            y = np.arange(len(frame))

            field = "increase_pct" if percentage else "net_increase"

            bars = ax.barh(y, frame[field], height=0.65, zorder=3, color=COLORS["nearest"])

            ax.bar_label(
                bars,

                padding   =6     ,
                fontweight="bold",

                labels=[
                    f"+{br(value)}{'%' if percentage else ''}"
                    for value in frame[field]
                ],
            )

            ax.set(
                yticks     =y             ,
                yticklabels=frame["label"],

                xlim=(0           , max(float(frame[field].max()) * 1.23, 1)),
                ylim=(len(y) - 0.5, -0.5                                    ),

                xlabel=(
                    "Acréscimo relativo às internações observadas (%)"
                    if   percentage
                    else "Acréscimo de internações (AIHs)"
                ),
            )

            if percentage:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: br(value)))
            else:
                ax.xaxis.set_major_locator  (MaxNLocator(integer=True))

            ax.xaxis.grid(True, alpha=0.18, zorder=0)

            ax.tick_params(axis="y", length=0)

            ax.spines[["left", "bottom"]].set_visible(False)

            detail.axis("off")
            detail.set (xlim=(0, 1), ylim=ax.get_ylim())

            detail.text(0.02, 1.03, "Observado → Cenário 1"                  , transform=detail.transAxes, weight="bold", fontsize=10)
            detail.text(0.88, 1.03, "Acréscimo" if percentage else "Variação", transform=detail.transAxes, weight="bold", fontsize=10, ha="center")

            for position, row in enumerate(frame.itertuples()):
                detail.text(
                    0.27    ,
                    position,

                    f"{br(row.observed_admissions)} → {br(row.nearest_admissions)}",

                    ha="center",
                    va="center",
                )

                value = f"+{br(row.net_increase)}" if percentage else f"+{br(row.increase_pct)}%"

                detail.text(0.88, position, value, ha="center", va="center", color="#52616D")

            fig.suptitle(title, x=0.04, y=0.97, fontsize=17, ha="left", weight="bold")

            fig.text(0.04, 0.915, subtitle, fontsize=11)
            fig.text(0.04, 0.025, footer  , fontsize=9 , color="#52616D")

            return fig

    return (provider_growth,)


@app.cell
def _(export_figure, impact, mo, provider_growth):
    _growth    , _growth_paths     = export_figure(provider_growth(impact["top_growth"    ]                 ), "provider_increase"           )
    _percentage, _percentage_paths = export_figure(provider_growth(impact["top_growth_pct"], percentage=True), "provider_percentage_increase")

    growth_paths = _growth_paths + _percentage_paths

    mo.vstack([_growth, _percentage])
    return (growth_paths,)


@app.cell
def _(
    COLORS,
    Figure,
    FuncFormatter,
    PLOT_STYLE,
    br,
    empty_figure,
    np,
    pd,
    plt,
):
    def provider_savings(top_saving: pd.DataFrame, saved_total: float) -> Figure:
        """Show national distance savings attributed to the selected new providers"""

        with plt.rc_context(PLOT_STYLE):
            title  = f"{len(top_saving)} provedores que mais contribuem para a economia de distância"
            footer = "SIH/SUS · Jan-jun/2026 · Frequência do fluxo observado x redução da distância por internação."

            if top_saving.empty or saved_total <= 0:
                return empty_figure(
                    (13.5, 8.8),
                    title      ,

                    "Nenhuma economia de distância disponível para este ranking.",

                    footer=footer,
                )

            fig, ax = plt.subplots(figsize=(13.5, 8.8))

            fig.subplots_adjust(
                left  =0.25,
                right =0.94,
                top   =0.83,
                bottom=0.11,
            )

            y = np.arange(len(top_saving))

            bars = ax.barh(
                y, top_saving["saved_admission_km"], color=COLORS["saving"], height=0.65, zorder=3,
            )

            ax.bar_label(
                bars,

                padding =6 ,
                fontsize=10,

                labels=[
                    f"{br(row.saved_admission_km, 1)}  ·  {br(row.saving_share_pct, 1)}%"
                    for row in top_saving.itertuples()
                ],
            )

            ax.set(
                yticks     =y                  ,
                yticklabels=top_saving["label"],

                xlim=(0           , max(float(top_saving["saved_admission_km"].max()) * 1.36, 1)),
                ylim=(len(y) - 0.5, -0.5                                                        ),

                xlabel="Economia ponderada atribuída ao novo provedor (km)",
            )

            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: br(value)))

            ax.xaxis.grid(True, alpha=0.18, zorder=0)

            ax.tick_params(axis="y", length=0)

            ax.spines[["left", "bottom"]].set_visible(False)

            fig.suptitle(title, x=0.04, y=0.97, fontsize=16, ha="left", weight="bold")

            share = 100 * top_saving["saved_admission_km"].sum() / saved_total

            fig.text(0.04, 0.91, f"Juntos acumulam {br(share, 1)}% da economia nacional", fontsize=11)
            fig.text(0.04, 0.025, footer                                                , fontsize=9 , color="#52616D")

            return fig

    return (provider_savings,)


@app.cell
def _(export_figure, impact, provider_savings, reductions):
    _figure, saving_paths = export_figure(
        provider_savings(
            impact    ["top_saving"                    ],
            reductions["absolute_distance_reduction_km"],
        ),

        "provider_distance_savings",
    )

    _figure
    return (saving_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MAPS OF NATIONAL RANKING CASES
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reuse IBGE boundaries and municipal centroids in EPSG:5880. Arrows illustrate origin–provider links; they do not trace the routes used to estimate distances. Local care is shown with a ring or schematic loop and retains zero distance.
    """)
    return


@app.cell
def _(COLORS):
    CAPITALS = {
        "RO" : ("1100205" , "Porto Velho"   ),
        "AC" : ("1200401" , "Rio Branco"    ),
        "AM" : ("1302603" , "Manaus"        ),
        "RR" : ("1400100" , "Boa Vista"     ),
        "PA" : ("1501402" , "Belém"         ),
        "AP" : ("1600303" , "Macapá"        ),
        "TO" : ("1721000" , "Palmas"        ),
        "MA" : ("2111300" , "São Luís"      ),
        "PI" : ("2211001" , "Teresina"      ),
        "CE" : ("2304400" , "Fortaleza"     ),
        "RN" : ("2408102" , "Natal"         ),
        "PB" : ("2507507" , "João Pessoa"   ),
        "PE" : ("2611606" , "Recife"        ),
        "AL" : ("2704302" , "Maceió"        ),
        "SE" : ("2800308" , "Aracaju"       ),
        "BA" : ("2927408" , "Salvador"      ),
        "MG" : ("3106200" , "Belo Horizonte"),
        "ES" : ("3205309" , "Vitória"       ),
        "RJ" : ("3304557" , "Rio de Janeiro"),
        "SP" : ("3550308" , "São Paulo"     ),
        "PR" : ("4106902" , "Curitiba"      ),
        "SC" : ("4205407" , "Florianópolis" ),
        "RS" : ("4314902" , "Porto Alegre"  ),
        "MS" : ("5002704" , "Campo Grande"  ),
        "MT" : ("5103403" , "Cuiabá"        ),
        "GO" : ("5208707" , "Goiânia"       ),
        "DF" : ("5300108" , "Brasília"      ),
    }

    MAP_DESTINATIONS = [
        ("observed_destination_code", COLORS["observed"], "--"),
        ("nearest_destination_code" , COLORS["nearest" ], "-" ),
    ]

    MAP_LABEL_POSITIONS = {
        ("Bom Jesus da Lapa/BA", "origin_code"             ) : (-12,   0, "right", "center"),
        ("Bom Jesus da Lapa/BA", "nearest_destination_code") : ( 12, -12, "left" , "top"   ),

        ("Guaratinga/BA", "origin_code"             ) : (-14,  3, "right", "bottom"),
        ("Guaratinga/BA", "nearest_destination_code") : ( 12, -8, "left" , "top"   ),

        ("Mirangaba/BA", "origin_code"             ) : (  8, 12, "left" , "bottom"),
        ("Mirangaba/BA", "nearest_destination_code") : (-10,  2, "right", "bottom"),

        ("José Gonçalves de Minas/MG", "origin_code"              ) : ( 10, -2, "left" , "center"),
        ("José Gonçalves de Minas/MG", "nearest_destination_code" ) : (-10,  5, "right", "bottom"),
        ("José Gonçalves de Minas/MG", "observed_destination_code") : (-10, -5, "right", "top"   ),

        ("Ninheira/MG", "origin_code"              ) : ( 10, -12, "left" , "top"   ),
        ("Ninheira/MG", "observed_destination_code") : (-10,   0, "right", "center"),
        ("Ninheira/MG", "nearest_destination_code" ) : ( 10,  12, "left" , "bottom"),
    }

    MAP_FOOTNOTE = (
        "SIH/SUS · Jan-jun/2026 · Malhas IBGE · Distâncias da matriz de rotas; ligações esquemáticas entre centroides."
    )
    return CAPITALS, MAP_DESTINATIONS, MAP_FOOTNOTE, MAP_LABEL_POSITIONS


@app.cell
def _(ROOT, STATE_CODES, gpd, load_boundaries, municipal):
    cartography = {
        "states"    : load_boundaries(ROOT, "UF"       ),
        "municipal" : load_boundaries(ROOT, "municipio"),

        "points" : gpd.GeoDataFrame(
            municipal.copy(),

            crs     =4674,
            geometry=gpd.points_from_xy(
                municipal["longitude"],
                municipal["latitude" ],
            ),
        ).to_crs(5880),

        "uf_to_code" : {
            uf : code
            for code, uf in STATE_CODES.items()
        },
    }
    return (cartography,)


@app.cell
def _(CAPITALS, TOP_N, br, impact, municipal):
    def _select_map_cases(flow_ranking, top_weighted_flows, municipal, top_n=10):
        """Keep the original national, regional and mechanism selection rules"""

        regions = [
            {"name" : "Bahia"               , "ufs" : {"BA"      }, "origin_ufs" : {"BA"}, "stem": "cases_bahia"      },
            {"name" : "Minas Gerais e Bahia", "ufs" : {"MG", "BA"}, "origin_ufs" : {"MG"}, "stem": "cases_minas_bahia"},
        ]

        top = flow_ranking.head(top_n)

        for region in regions:
            mask = top["origin_state"].isin(region["origin_ufs"])

            for field in ("origin_state", "observed_state", "nearest_state"):
                mask &= top[field].isin(region["ufs"])

            region["cases"] = top.loc[mask].head(3).copy()

        capital_codes = {uf: item[0] for uf, item in CAPITALS.items()}

        capital      = flow_ranking["origin_state"].map(capital_codes)

        capital_mask = (
            flow_ranking["observed_destination_code"].eq(capital                     ) &
            flow_ranking["nearest_state"            ].eq(flow_ranking["origin_state"]) &
            flow_ranking["nearest_destination_code" ].ne(capital)
        )
        border_mask = (
            flow_ranking["observed_state"].eq(flow_ranking["origin_state"]) &
            flow_ranking["nearest_state" ].ne(flow_ranking["origin_state"])
        )

        weighted_order = [
            "saved_admission_km"       ,
            "distance_reduction_km"    ,
            "origin_code"              ,
            "observed_destination_code",
        ]

        def supplement(mask):
            return flow_ranking.loc[mask].sort_values(weighted_order, ascending=[False, False, True, True]).head(3).copy()

        groups = [
            {
                "cases": flow_ranking.head(4).copy(),

                "stem"           : "top4_flow_distance_maps"                                                  ,
                "title"          : "Quatro maiores reduções de deslocamento por internação"                   ,
                "subtitle"       : "Ranking nacional por redução em km"                                       ,
                "selection_rule" : "Top 4 by distance reduction, savings, origin and observed-provider codes.",
            },

            {
                "cases" : top_weighted_flows.head(4).copy(),

                "stem"           : "top4_flow_weighted_maps"                                           ,
                "title"          : "Quatro maiores economias ponderadas por fluxo"                     ,
                "subtitle"       : "Ranking nacional por internações × redução em km"                  ,
                "selection_rule" : "Top 4 by admission-weighted savings, distance reduction and codes.",
            },

            {
                "cases" : supplement(border_mask),

                "stem"           : "new_cross_state_cases"                                                                                           ,
                "title"          : "Travessias de fronteira estadual para um provedor mais próximo"                                                  ,
                "subtitle"       : f"Até 3 maiores economias entre {br(border_mask.sum())} fluxos elegíveis"                                         ,
                "selection_rule" : "Observed provider in the origin state; new provider across a state border; top 3 by savings, distance and codes.",
            },
        ]

        return {
            "regions" : regions,
            "groups"  : groups ,

            "metadata": {
                "national_cases_per_figure"      : 4,
                "supplementary_cases_per_figure" : 3,

                "eligible_capital_to_noncapital_flows" : int(capital_mask.sum()),
                "eligible_first_cross_border_flows"    : int(border_mask .sum()),

                "capital_to_noncapital_cases" : supplement(capital_mask).to_dict(orient="records"),

                "capital_reference" : [
                    {
                        "state"         : uf  ,
                        "capital_code"  : code,
                        "capital_label" : f"{name}/{uf}",
                    }
                    for uf, (code, name) in CAPITALS.items()
                    if uf in set(municipal["state"])
                ],

                "local_arrow" : "Schematic loop for same-municipality care; modeled distance remains zero.",
            },
        }

    map_cases = _select_map_cases(
        impact["flow_ranking"      ],
        impact["top_weighted_flows"],

        municipal,

        top_n=TOP_N,
    )
    return (map_cases,)


@app.cell
def _(COLORS, FancyArrowPatch, Line2D, MAP_DESTINATIONS, MAP_FOOTNOTE):
    def state_labels(ax, mesh, ufs, uf_to_code, fontsize, limits=None):
        for uf in sorted(ufs):
            center = mesh.loc[uf_to_code[uf]].geometry.representative_point()

            if limits and not (limits[0][0] < center.x < limits[0][1] and limits[1][0] < center.y < limits[1][1]):
                continue

            ax.text(center.x, center.y, uf, zorder=1, fontsize=fontsize, color="#CCD5DC", weight="bold", ha="center", va="center")

    def map_arrows(ax, row, points, span=None):
        origin = points.loc[row.origin_code].geometry

        for field, color, style in MAP_DESTINATIONS:
            destination = points.loc[getattr(row, field)].geometry

            if getattr(row, field) == row.origin_code:
                if span is not None:
                    radius = 0.055 * span

                    ax.add_patch(
                        FancyArrowPatch(
                            (origin.x - radius, origin.y),
                            (origin.x, origin.y + radius),

                            lw            =2.2,
                            zorder        =4  ,
                            mutation_scale=14 ,
                            color         =color,
                            linestyle     =style,

                            arrowstyle     ="-|>"          ,
                            connectionstyle="arc3,rad=-1.2",
                        )
                    )

                ax.scatter(origin.x, origin.y, s=165 if span else 180, facecolors="none", edgecolors=color, lw=2, zorder=4 if span else 6)
            else:
                ax.annotate(
                    "",

                    xy    =(destination.x, destination.y),
                    xytext=(origin     .x, origin     .y),

                    zorder=3,

                    arrowprops={
                        "arrowstyle" : "-|>",
                        "color"      : color,
                        "linestyle"  : style,

                        "shrinkA" : 4,
                        "shrinkB" : 5,

                        "lw"             : 2.2 if span else 2 ,
                        "mutation_scale" : 14  if span else 12,
                    }
                )

    def map_legend(fig, regional=False):
        handles = [
            Line2D([], [], color=COLORS["observed"], marker="s", ls="--"  , label="Provedor observado" if regional else "Observado"            ),
            Line2D([], [], color=COLORS["nearest" ], marker="^",            label="Novo provedor"      if regional else "Otimizado (cenário 1)"),
            Line2D([], [], color=COLORS["text"    ], marker="o", ls="none", label="Origem"                                                     ),
        ]

        if regional:
            handles.append(Line2D([], [], color=COLORS["nearest"], marker="o", ls="none", label="Atendimento local", markerfacecolor="none"))

        fig.legend(
            handles=handles     ,
            ncol   =len(handles),

            fontsize=9 if regional else 10,

            frameon       =False         ,
            loc           ="lower center",
            bbox_to_anchor=(0.5, 0.042 if regional else 0.039),
        )

        fig.text(0.5, 0.015, MAP_FOOTNOTE, fontsize=9, ha="center", color="#52616D")

    return map_arrows, map_legend, state_labels


@app.cell
def _(
    COLORS,
    MAP_LABEL_POSITIONS,
    PLOT_STYLE,
    br,
    map_arrows,
    map_legend,
    plt,
    state_labels,
):
    def regional_case_maps(cases, region, context, top_n=10):
        """Draw the original equal-extent Bahia or Minas Gerais/Bahia panels"""

        if cases.empty:
            return None

        mesh   = context["states"].loc[[context["uf_to_code"][uf] for uf in sorted(region["ufs"])]]
        points = context["points"]

        xmin, ymin, xmax, ymax = mesh.total_bounds

        padding = 0.06 * max(xmax - xmin, ymax - ymin)

        limits = ((xmin - padding, xmax + padding), (ymin - padding, ymax + padding))

        with plt.rc_context(PLOT_STYLE):
            fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 8.5), squeeze=False)

            fig.subplots_adjust(left=0.015, right=0.985, bottom=0.20, top=0.77, wspace=0.10)

            for ax, row in zip(axes.ravel(), cases.itertuples()):
                mesh.plot(ax=ax, facecolor="#F2F5F7", edgecolor="#AAB6C0", linewidth=0.8)

                state_labels(ax, mesh, region["ufs"], context["uf_to_code"], 24)

                map_arrows(ax, row, points)

                for field, label, color, marker in [
                    ("origin_code"              , row.origin_label  , COLORS["text"    ], "o"),
                    ("observed_destination_code", row.observed_label, COLORS["observed"], "s"),
                    ("nearest_destination_code" , row.nearest_label , COLORS["nearest" ], "^"),
                ]:
                    point = points.loc[getattr(row, field)].geometry

                    ax.scatter(
                        point.x,
                        point.y,

                        zorder    =5  ,
                        s         =40 ,
                        linewidths=0.6,

                        marker=marker,
                        color =color ,

                        edgecolors="white",
                    )

                    dy = -13 if field == "origin_code" else 10

                    dx, dy, ha, va = MAP_LABEL_POSITIONS.get(
                        (row.origin_label, field), (0, dy, "center", "top" if dy < 0 else "bottom"),
                    )

                    label = label.replace("José Gonçalves de Minas", "José Gonçalves\nde Minas").replace("/", "\n")

                    ax.annotate(
                        label             ,
                        (point.x, point.y),

                        xytext    =(dx, dy)       ,
                        textcoords="offset points",

                        fontsize=8,
                        zorder  =7,

                        ha   =ha,
                        va   =va,
                        color=color,

                        bbox={
                            "facecolor" : "white",
                            "edgecolor" : "none" ,

                            "alpha" : 0.8,
                            "pad"   : 1.5,
                        }
                    )

                ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")

                ax.set_axis_off()

                ax.set_title(row.origin_label, fontsize=11, pad=12)

                ax.text(
                    0.5   ,
                    -0.065,

                    f"{br(row.observed_distance_km, 1)} → {br(row.nearest_distance_km, 1)} km\n"
                    f"-{br(row.distance_reduction_km, 1)} km por internação ({br(row.distance_reduction_pct, 1)}%)\n"
                    f"{br(row.observed_flow)} AIH(s) · {br(row.saved_admission_km, 1)} internações·km evitados",

                    fontsize   =10 ,
                    linespacing=1.6,

                    ha       ="center",
                    va       ="top"   ,
                    transform=ax.transAxes,
                )

            fig.suptitle(f"Deslocamentos alterados · {region['name']}", x=0.025, y=0.98, fontsize=17, ha="left", weight="bold")

            fig.text(0.025, 0.915, f"{len(cases)} casos do top {top_n} nacional", fontsize=11)

            map_legend(fig, regional=True)

        return fig

    return (regional_case_maps,)


@app.cell
def _(TOP_N, cartography, export_figure, map_cases, mo, regional_case_maps):
    regional_map_paths = []
    _panels            = []

    for _region in map_cases["regions"]:
        _figure, _paths = export_figure(
            regional_case_maps(
                _region["cases"],
                _region         ,
                cartography     ,

                top_n=TOP_N,
            ),

            _region["stem"],
        )

        regional_map_paths.extend(_paths)

        _panels.append(
            _figure
            if _figure is not None
            else mo.md(
                f"No national top-{TOP_N} flows meet the selection for **{_region['name']}**."
            )
        )

    mo.vstack(_panels)
    return (regional_map_paths,)


@app.cell
def _(
    COLORS,
    PLOT_STYLE,
    br,
    fill,
    map_arrows,
    map_legend,
    np,
    plt,
    state_labels,
):
    def selected_case_maps(cases, title, subtitle, context, *, stem="selected_cases", selection_rule=""):
        """Draw individually scaled case panels and return JSON-ready provenance"""

        metadata = {
            "stem"           : stem          ,
            "selection_rule" : selection_rule,

            "n_cases"       : len(cases) ,
            "shared_extent" : False      ,
            "crs"           : "EPSG:5880",

            "panel_extents" : [],
        }

        if cases.empty:
            return None, metadata

        points, states = context["points"], context["states"]

        with plt.rc_context(PLOT_STYLE):
            fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 8.6), squeeze=False)

            fig.subplots_adjust(left=0.025, right=0.985, top=0.80, bottom=0.285, wspace=0.12)

            for position, (ax, row) in enumerate(zip(axes.ravel(), cases.itertuples()), start=1):
                codes = [
                    row.origin_code              ,
                    row.observed_destination_code,
                    row.nearest_destination_code ,
                ]

                coordinates = points.loc[codes].geometry

                xs, ys = coordinates.x.to_numpy(), coordinates.y.to_numpy()

                span = max(float(np.ptp(xs)), float(np.ptp(ys)), 30_000.0)

                padding = 0.19 * span

                box = ax.get_position(original=True)

                ratio = (box.width * fig.get_figwidth()) / (box.height * fig.get_figheight())

                width  = max(float(np.ptp(xs)) + 2 * padding, (float(np.ptp(ys)) + 2 * padding) * ratio)
                height = width / ratio

                midx, midy = float((xs.min() + xs.max()) / 2)    , float((ys.min() + ys.max()) / 2)
                xlim, ylim = (midx - width / 2, midx + width / 2), (midy - height / 2, midy + height / 2)

                bounds  = states.bounds
                visible = states.loc[
                    (bounds["maxx"] >= xlim[0]) & (bounds["minx"] <= xlim[1]) &
                    (bounds["maxy"] >= ylim[0]) & (bounds["miny"] <= ylim[1])
                ]

                visible.plot(ax=ax, zorder=0, linewidth=0.8, facecolor="#F2F5F7", edgecolor="#AAB6C0")

                state_labels(
                    ax, states, {row.origin_state, row.observed_state, row.nearest_state}, context["uf_to_code"], 21, (xlim, ylim)
                )

                map_arrows(ax, row, points, span)

                origin = points.loc[row.origin_code].geometry

                local  = row   .nearest_destination_code == row.origin_code

                for field, role, color, marker, offset, ha, va in [
                    ("origin_code"              , "Origem"                                , COLORS["text"    ], "o", (-8, -10), "right", "top"   ),
                    ("observed_destination_code", "Observado"                             , COLORS["observed"], "s", (-7,  10), "right", "bottom"),
                    ("nearest_destination_code" , "Local · 0 km" if local else "Otimizado", COLORS["nearest" ], "^", ( 8,  10), "left" , "bottom"),
                ]:
                    point = points.loc[getattr(row, field)].geometry

                    if not local:
                        other = points.loc[row.nearest_destination_code].geometry if field == "origin_code" else origin

                        dx, dy = (8 if point.x >= other.x else -8), (10 if point.y >= other.y else -10)

                        offset, ha, va = (dx, dy), "left" if dx > 0 else "right", "bottom" if dy > 0 else "top"

                    ax.scatter (point.x, point.y, s=47, zorder=5, linewidths=0.7, marker=marker, color=color, edgecolors="white")

                    ax.annotate(
                        role              ,
                        (point.x, point.y),

                        fontsize=9,
                        zorder  =7,

                        xytext=offset,
                        color =color ,
                        va    =va,
                        ha    =ha,

                        textcoords="offset points",

                        bbox={
                            "facecolor" : "white",
                            "edgecolor" : "none" ,
                            "alpha" : 0.9,
                            "pad"   : 1.5,
                        }
                    )

                ax.set(xlim=xlim, ylim=ylim, aspect="equal")

                ax.set_axis_off()

                ax.set_title(fill(row.origin_label, width=28), fontsize=10.5, pad=14)

                center = (box.x0 + box.x1) / 2

                fig.text(
                    center,
                    0.25  ,

                    f"{fill(f'Observado: {row.observed_label}', width=37)}\n"
                    f"{fill(f'Otimizado: {row.nearest_label}', width=37)}\n"
                    f"{br(row.observed_distance_km, 1)} → {br(row.nearest_distance_km, 1)} km\n"
                    f"-{br(row.distance_reduction_km, 1)} km por AIH ({br(row.distance_reduction_pct, 1)}%)\n"
                    f"{br(row.observed_flow)} AIH(s) · {br(row.saved_admission_km, 1)} int.·km evitados",

                    ha="center",
                    va="top"   ,

                    fontsize   =9.5 ,
                    linespacing=1.55,
                )

                metadata["panel_extents"].append(
                    {
                        "map_group" : stem    ,
                        "panel"     : position,

                        **dict(
                            zip(
                                [
                                    "origin_code"              ,
                                    "observed_destination_code",
                                    "nearest_destination_code" ,
                                ],

                                codes
                            )
                        ),

                        "xmin" : xlim[0],
                        "xmax" : xlim[1],
                        "ymin" : ylim[0],
                        "ymax" : ylim[1],
                        "crs"  : "EPSG:5880",
                    }
                )

            fig.suptitle(title, x=0.025, y=0.98, fontsize=17, ha="left", weight="bold")

            fig.text(0.025, 0.918, subtitle, fontsize=11)

            map_legend(fig)

        return fig, metadata

    return (selected_case_maps,)


@app.cell
def _(cartography, export_figure, map_cases, mo, selected_case_maps):
    selected_map_paths    = []
    selected_map_metadata = []

    _panels = []

    for _group in map_cases["groups"]:
        _figure, _metadata = selected_case_maps(
            _group["cases"   ],
            _group["title"   ],
            _group["subtitle"],

            cartography,

            stem          =_group["stem"          ],
            selection_rule=_group["selection_rule"],
        )

        _figure, _paths = export_figure(_figure, _group["stem"])

        selected_map_paths   .extend(_paths   )
        selected_map_metadata.append(_metadata)

        _panels.append(
            _figure
            if   _figure is not None
            else mo.md(
                f"No eligible flows for `{_group['stem']}`."
            )
        )

    mo.vstack(_panels)
    return (selected_map_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## REGIONAL VARIANT: AMAZONAS AND PARÁ
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Retain observed flows with **both origin and destination in AM or PA**. Their admissions define the regional demand, which is reallocated only to observed provider municipalities within AM or PA. Residents treated elsewhere are excluded from both panels. This is a separate regional scenario, rather than a geographic subset of the national solution.
    """)
    return


@app.cell
def _(
    CASE_STATES,
    compare_metrics,
    municipal,
    nearest_provider,
    observed,
    od,
    scenario_metrics,
):
    _states   = municipal["state"]

    _regional = _states.isin(CASE_STATES)

    case_observed = observed.loc[
        observed["origin_code"     ].map(_regional) &
        observed["destination_code"].map(_regional)
    ].copy()

    _demand = case_observed.groupby("origin_code")["observed_flow"].sum()

    _providers = municipal.index[_regional & municipal["observed_supply"].gt(0)]

    _od = od.loc[
        od["origin_code"     ].isin(_demand.index) &
        od["destination_code"].isin(_providers   )
    ]

    case_observed.attrs["n_available_providers"] = len(_providers)

    case_scenario = nearest_provider(_od, _demand)
    case_metrics  = {
        "observed"         : scenario_metrics(case_observed, "observed_flow" ),
        "nearest_provider" : scenario_metrics(case_scenario, "allocated_flow"),
    }

    case_metrics["comparison"] = compare_metrics(
        case_metrics["observed"        ],
        case_metrics["nearest_provider"],
    )
    return case_metrics, case_observed, case_scenario


@app.cell
def _(
    COLORS,
    Line2D,
    LineCollection,
    MAP_FOOTNOTE,
    PLOT_STYLE,
    br,
    np,
    pd,
    plt,
):
    def regional_comparison(observed, scenario, municipal, context):
        """Plot the separately recalculated AM/PA case, retaining a common scale"""

        if observed.empty or scenario.empty:
            return None

        municipal = municipal.set_index("municipality_code") if "municipality_code" in municipal else municipal

        state_codes = [
            context["uf_to_code"][uf]
            for uf in ("AM", "PA")
        ]

        states     = context["states"   ].loc[state_codes]
        boundaries = context["municipal"].loc[context["municipal"].index.str[:2].isin(state_codes)]
        points     = context["points"   ]

        xy = np.column_stack((points.geometry.x, points.geometry.y))

        coordinates = pd.DataFrame(xy, index=points.index)

        provider_codes = municipal.index[
            municipal["state"          ].isin(["AM", "PA"]) &
            municipal["observed_supply"].gt  (0)
        ]

        providers = coordinates.loc[provider_codes].to_numpy()

        xmin, ymin, xmax, ymax = states.total_bounds

        padding = 0.025 * max(xmax - xmin, ymax - ymin)

        frequency = "allocated_flow" if "allocated_flow" in scenario else "case_allocated_flow"

        before = float((observed["observed_flow"] * observed["distance_km"]).sum())
        after  = float((scenario[frequency      ] * scenario["distance_km"]).sum())

        reduction_pct = 100 * (before - after) / before if before else 0.0

        with plt.rc_context(PLOT_STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), sharex=True, sharey=True)

            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.19, top=0.84, wspace=0.05)

            for ax, flows, count, title, total in [
                (axes[0], observed, "observed_flow", "A) Fluxo observado"      , before),
                (axes[1], scenario, frequency      , "B) Provedor mais próximo", after ),
            ]:
                boundaries.plot(ax=ax, facecolor="#F6F7F8", edgecolor="#CFD5D9", linewidth=0.22)

                states.boundary.plot(ax=ax, color="#79868F", linewidth=0.8)

                origins      = coordinates.loc[flows["origin_code"     ]].to_numpy()
                destinations = coordinates.loc[flows["destination_code"]].to_numpy()

                volumes = flows[count        ]                              .to_numpy(dtype="int64")
                local   = flows["origin_code"].eq(flows["destination_code"]).to_numpy(dtype=bool   )

                ax.add_collection(LineCollection(
                    np.stack([origins[~local], destinations[~local]], axis=1),

                    zorder    =3  ,
                    alpha     =0.6,
                    colors    =COLORS["nearest"]                    ,
                    linewidths=0.4 + 0.65 * np.sqrt(volumes[~local]),
                ))

                ax.scatter(origins  [~local, 0], origins  [~local, 1], s=9                                , color=COLORS["nearest"], zorder=4)
                ax.scatter(providers[:     , 0], providers[:     , 1], s=24                               , color="#182C3A"        , zorder=5)
                ax.scatter(
                    origins[local, 0],
                    origins[local, 1],

                    zorder    =6  ,
                    linewidths=1.1,

                    s         =40 + 24 * np.sqrt(volumes[local]),
                    facecolors="none"           ,
                    edgecolors=COLORS["nearest"],
                )

                ax.set      (xlim=(xmin - padding, xmax + padding), ylim=(ymin - padding, ymax + padding), aspect="equal")

                ax.set_title(f"{title}\nDistância total = {br(total, 1)} km", loc="left", fontsize=11, linespacing=1.8, pad=14, y=1.03)

                ax.set_axis_off()

            fig.suptitle("Fluxo observado e realocação para provedor mais próximo · Amazonas e Pará", fontsize=15, y=0.96)

            fig.legend(
                handles=[
                    Line2D([], [], marker="o", linestyle="none", color="#182C3A"       , markersize=np.sqrt(24), label="Município provedor"),
                    Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markersize=9, markeredgecolor=COLORS["nearest"], label="Admissões locais"),
                ],

                loc           ="lower center",
                bbox_to_anchor=(0.5, 0.1)    ,

                ncol    =4,
                fontsize=9,
                frameon =False,
            )

            fig.text(0.5, 0.065, f"Redução de distância: {br(before - after, 1)} km ({br(reduction_pct, 1)}%)", ha="center", fontsize=10)
            fig.text(0.5, 0.025, MAP_FOOTNOTE                                                                 , ha="center", fontsize=8, color="#53616A")

        return fig

    return (regional_comparison,)


@app.cell
def _(
    cartography,
    case_observed,
    case_scenario,
    export_figure,
    municipal,
    regional_comparison,
):
    _figure, case_paths = export_figure(
        regional_comparison(
            case_observed,
            case_scenario,
            municipal    ,
            cartography  ,
        ),

        "am_pa_observed_vs_nearest_provider",
    )

    _figure
    return (case_paths,)


@app.cell
def _(case_metrics, mo, pd):
    mo.ui.table(
        pd.DataFrame(
            {
                key : case_metrics[key]
                for key in ("observed", "nearest_provider")
            }
        )
        .rename_axis("metric")
        .reset_index(),

        selection=None,
        label    ="AM/PA regional metrics (same retained admissions in both scenarios)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RESULTS AND PROVENANCE
    """)
    return


@app.cell
def _(
    METRICS,
    MUNICIPAL,
    OD,
    RESULT,
    ROOT,
    case_paths,
    comparison_paths,
    growth_paths,
    input_sources,
    json,
    metrics,
    mo,
    reductions,
    regional_map_paths,
    saving_paths,
    scenario,
    selected_map_paths,
    sha256,
    write_csv,
):
    _figures = comparison_paths + growth_paths + saving_paths + regional_map_paths + selected_map_paths + case_paths

    _sources = [
        ROOT / ".cache/ibge" / name
        for name in (
            "malha_estados_brasil_minima.geojson"   ,
            "malha_municipios_brasil_minima.geojson",
        )
    ]

    _inputs = {
        str(p.relative_to(ROOT)) : sha256(p.read_bytes()).hexdigest()
        for p in _sources
    }

    for _name, _path in (("od", OD), ("municipal", MUNICIPAL)):
        _digest = input_sources[_name]["sha256"]

        if sha256(_path.read_bytes()).hexdigest() != _digest:
            raise ValueError("Inputs changed during execution; reload the data before exporting.")

        _inputs[str(_path.relative_to(ROOT))] = _digest

    write_csv(scenario, RESULT)

    _payload = {
        "scenario"      : 1                                            ,
        "scenario_name" : "unconstrained_nearest_provider_reallocation",

        "metrics" : {
            "nearest_provider" : metrics   ,
            "comparison"       : reductions,
        },

        "outputs" : {
            str(p.relative_to(ROOT)) : sha256(p.read_bytes()).hexdigest()
            for p in [RESULT, *_figures]
        },
    }


    METRICS.parent.mkdir(parents=True, exist_ok=True)

    METRICS.write_text(
        json.dumps(
            _payload,

            ensure_ascii=False,
            allow_nan   =False,
            indent      =2    ,
        ) + "\n",

        encoding="utf-8",
    )

    mo.md(
        f"**Validated and exported:** `{RESULT.relative_to(ROOT)}`; "
        f"`{METRICS.relative_to(ROOT)}`; {len(_figures) // 2} figures (PDF)."
    )
    return


if __name__ == "__main__":
    app.run()
