import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # HISTORICAL SURGICAL FLOWS AND CARBON
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **2016–2025 · Oncological prostatectomy · Brazil · SIH/SUS**

    Each year/procedure keeps its own origin demand, observed provider municipalities and production. Admissions are regular AIHs per facility, not unique patients; observed production is the capacity proxy.

    All years use the same **Saldanha May 2024 reference network**, including the shared route corrections. Model costs use `numpy.rint` integer kilometres; metrics and carbon use the original decimal distances. Centroids are retained only for municipal objects and maps.
    """)
    return


@app.cell
def _():
    import sys

    import marimo as mo

    from pathlib import Path

    return Path, mo, sys


@app.cell
def _(Path, sys):
    ROOT = next(
        path
        for path in (
             Path.cwd()        ,
            *Path.cwd().parents,
        )
        if (path / "pyproject.toml").is_file()
    )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return (ROOT,)


@app.cell
def _():
    from optimization.history import comparison_table

    from src.analysis import DEFAULT_CARBON, export_history, load_history, run_history

    return (
        DEFAULT_CARBON,
        comparison_table,
        export_history,
        load_history,
        run_history,
    )


@app.cell
def _(DEFAULT_CARBON):
    YEARS             = tuple(range(2016, 2026))
    PROCESSING_CUTOFF = "2026-09"

    PROCEDURES = ("0416010121",)

    WORKERS = 4
    CARBON  = DEFAULT_CARBON.copy()

    CACHED_RUN = None
    return CACHED_RUN, CARBON, PROCEDURES, PROCESSING_CUTOFF, WORKERS, YEARS


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button(label="Acquire/reuse SIH snapshot and analyze years")

    mo.vstack([
        run_button,

        mo.md(
            "> **Note**.: Results stay in `.cache/history/`. The first acquisition reads thousands of RD files and later runs reuse."
        ),
    ])
    return (run_button,)


@app.cell
def _(
    CACHED_RUN,
    CARBON,
    PROCEDURES,
    PROCESSING_CUTOFF,
    ROOT,
    WORKERS,
    YEARS,
    load_history,
    mo,
    run_button,
    run_history,
):
    mo.stop(
        not run_button.value and CACHED_RUN is None,

        mo.md(
            "Run the analysis above, or set `CACHED_RUN` to inspect a completed run."
        )
    )

    if CACHED_RUN is not None:
        result = load_history(CACHED_RUN, carbon=CARBON)
    else:
        result = run_history (
            ROOT      ,
            YEARS     ,
            PROCEDURES,

            processing_cutoff=PROCESSING_CUTOFF,
            workers          =WORKERS          ,
            carbon           =CARBON           ,
        )
    return (result,)


@app.cell(hide_code=True)
def _(comparison_table, mo, result):
    mo.vstack([
        mo.md("## COMPARISON"),
        mo.ui.table(comparison_table(result.summary), selection=None),

        mo.md(
            "Infeasible annual scenarios are retained. A decade summary is incomplete when any requested year/procedure is missing."
        ),

        mo.md("## COVERAGE"),
        mo.ui.table(result.coverage                 , selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo, result):
    mo.vstack([
        mo.md(r"""
        ## POTENTIAL TRANSPORT EMISSIONS

        Avoided tCO₂e = saved km × distance multiplier × legs × travelers ×
        emission factor / 1,000. The default is 0.02776 kgCO₂e/passenger-km,
        attributed to DESNZ 2025 Coach, with a round trip for one traveler.
        This fixed factor is a scenario assumption, not a measurement of the
        Brazilian fleet in each year. It excludes surgical infrastructure and
        does not represent certified carbon credits. No monetary value is
        calculated unless a price per tonne is provided.
        """),

        mo.ui.table(
            result.summary[
                [
                    "scenario"       ,
                    "status"         ,
                    "transport_tco2e",
                    "avoided_tco2e"  ,
                    "potential_value",
                ]
            ],

            selection=None,
        ),

        mo.md("## ANNUAL STATISTICS"),
        mo.ui.table(result.annual, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo, result):
    export_button = mo.ui.run_button(label="Export consolidated tables and HTML report")

    mo.vstack([
        mo.md(f"Completed artifacts: `{result.directory}`"),

        export_button,
    ])
    return (export_button,)


@app.cell
def _(ROOT, export_button, export_history, mo, result):
    mo.stop(not export_button.value)

    folder, report = export_history(result, ROOT)

    mo.vstack([
        mo.md(f"Tables and manifest: `{folder}`."),
        mo.md(f"Report: `{report}`."             ),
    ])
    return


if __name__ == "__main__":
    app.run()
