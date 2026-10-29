import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CAPACITY-CONSTRAINED REALLOCATION
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Scenario 2 · Oncological prostatectomy · Brazil · January–June 2026.**

    Reallocate the observed admissions among the same provider municipalities. Observed production is a capacity proxy; expansion is hypothetical and applied to each provider separately.
    """)
    return


@app.cell
def _():
    import sys
    import json

    from pathlib            import Path
    from hashlib            import sha256
    from importlib.metadata import version
    from datetime           import datetime, timezone

    import numpy             as np
    import pandas            as pd
    import marimo            as mo
    import pyomo.environ     as pyo
    import matplotlib.pyplot as plt

    return Path, datetime, json, mo, np, pd, plt, pyo, sha256, sys, timezone


@app.cell
def _(Path, sys):
    ROOT = next(
        path
        for path in (
             Path.cwd()        ,
            *Path.cwd().parents,
        )
        if (path / "data/telesurgery_2026_first_half.txt").is_file()
    )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return (ROOT,)


@app.cell
def _():
    from optimization.data             import load_instance
    from optimization.scenarios        import TransportScenario
    from optimization.models.transport import build_transport_model

    from optimization.metrics    import flow_metrics
    from optimization.validation import validate_solution
    from optimization.solve      import extract_solution, solve_model, solver_options

    return (
        TransportScenario,
        build_transport_model,
        extract_solution,
        flow_metrics,
        load_instance,
        solve_model,
        solver_options,
        validate_solution,
    )


@app.cell
def _():
    PERIOD = "2026_first_half"

    SLACK_PCT      = (*range(26) , *range(30, 101, 5))
    FLOW_SLACK_PCT = (0, 1, 2, 5, 10, 15, 25, 50, 100)
    return FLOW_SLACK_PCT, PERIOD, SLACK_PCT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MODEL AND DISTANCES
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For origin demand $D_i$, observed provider production $U_j^{obs}$ and slack $s$:

    $$
    \min_x \sum_{i,j}\tilde d_{ij}x_{ij},\qquad
    \sum_j x_{ij}=D_i,\qquad
    \sum_i x_{ij}\leq\lfloor(1+s)U_j^{obs}\rfloor,\qquad x_{ij}\geq0.
    $$

    The compact instance supplies demand, production and **rounded integer kilometres** $\tilde d_{ij}=\lfloor d_{ij}\rfloor$. The transportation LP has integral margins; HiGHS simplex yields an integral basic solution, checked before export.
    """)
    return


@app.cell
def _(PERIOD, ROOT, load_instance, pd):
    data = load_instance(ROOT / "data", PERIOD)

    od_path      = ROOT / f"data/origin_destination_data_{PERIOD}.csv"
    nearest_path = ROOT / f"results/scenario_1_{PERIOD}.csv"

    _codes = {
        "origin_code"      : "string",
        "destination_code" : "string",
    }

    _keys  = list(_codes)

    od = pd.read_csv(
        od_path,

        dtype  =_codes,
        usecols=[
            *_keys,

            "distance_km"  ,
            "observed_flow",
            "same_state"   ,
        ],
    )

    distances = od.set_index(_keys).distance_km

    observed  = od.loc[od.observed_flow.gt(0)].rename(
        columns={
            "observed_flow" : "allocated_flow"
        }
    ).copy()

    observed["same_municipality"] = observed.origin_code.eq(observed.destination_code)
    observed["total_distance_km"] = observed.allocated_flow * observed.distance_km

    nearest = pd.read_csv(nearest_path, dtype=_codes)

    observed_total = float(observed.total_distance_km.sum   ())
    n_admissions   = sum  (data    .demand           .values())
    return (
        data,
        distances,
        n_admissions,
        nearest,
        nearest_path,
        observed,
        observed_total,
        od_path,
    )


@app.cell
def _(
    FLOW_SLACK_PCT,
    SLACK_PCT,
    TransportScenario,
    build_transport_model,
    data,
    distances,
    extract_solution,
    flow_metrics,
    mo,
    observed_total,
    pd,
    pyo,
    solve_model,
    solver_options,
    validate_solution,
):
    _rows, _selected = [], []

    for _pct in mo.status.progress_bar(
        SLACK_PCT,

        title="Solving capacity scenarios",
    ):
        _scenario = TransportScenario(slack=_pct / 100)

        _model              = build_transport_model(data  , _scenario)
        used_solver_options = solver_options       (_model)

        _result = solve_model(_model)

        _objective = float(pyo.value(_model.total_distance))

        _solution = extract_solution (_model   , data, distances=distances  )
        _checks   = validate_solution(_solution, data, _scenario, _objective)

        _rows.append({
            "scenario"  : f"{_pct}%"     ,
            "slack"     : _scenario.slack,
            "slack_pct" : _pct           ,

            "total_capacity" : sum(
                _scenario.capacity(supply)
                for supply in data.observed_supply.values()
            ),

            "model_objective_km" : _objective                               ,
            "termination"        : str(_result.solver.termination_condition),

            **flow_metrics(_solution, observed_total),

            **_checks,
        })

        if _pct in FLOW_SLACK_PCT:
            _selected.append(_solution.assign(slack=_scenario.slack))

        del _model, _result


    scenario_metrics = pd.DataFrame(_rows)

    flows = pd.concat(_selected, ignore_index=True)
    flows = flows[
        [
            "slack",

            *[
                name
                for name in flows.columns
                if  name != "slack"
            ]
        ]
    ]

    if scenario_metrics.model_objective_km.diff().dropna().gt(1e-5).any():
        raise RuntimeError("The optimized integer objective increased with capacity.")
    return flows, scenario_metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GAINS AND DESCRIPTIVE ELBOW
    """)
    return


@app.cell
def _(np):
    def find_elbow(slacks, totals, tolerance=1e-5):
        """Maximum normalized chord deviation, with the earliest tied level"""

        x = np.asarray(slacks)
        y = np.asarray(totals)

        span = y[0] - y[-1]

        if span <= tolerance:
            return None, np.full(len(x), np.nan)

        scores          = (y[0] - y) / span - (x - x[0]) / (x[-1] - x[0])
        score_tolerance = max(1e-12, 2 * tolerance / span)

        peak = scores[1:-1].max()

        if peak <= score_tolerance:
            return None, scores

        index = np.flatnonzero(
            np.isclose(
                scores[1:-1],
                peak        ,

                rtol=0              ,
                atol=score_tolerance,
            )
        )[0] + 1

        return int(index), scores

    return (find_elbow,)


@app.cell
def _(find_elbow, n_admissions, np, scenario_metrics):
    sensitivity = scenario_metrics.copy()

    sensitivity["effective_expansion_pct"] = 100 * (sensitivity.total_capacity / n_admissions - 1)
    sensitivity["capacity_increment"     ] =        sensitivity.total_capacity   .diff()
    sensitivity["slack_increment_pp"     ] =        sensitivity.slack_pct        .diff()
    sensitivity["absolute_gain_km"       ] =       -sensitivity.total_distance_km.diff()
    sensitivity["relative_gain_pct"      ] = (
        100 * sensitivity.absolute_gain_km / sensitivity.total_distance_km.shift()
    )

    sensitivity["mean_gain_km"              ] = sensitivity.absolute_gain_km         / n_admissions
    sensitivity["marginal_total_km_per_pp"  ] = sensitivity.absolute_gain_km         / sensitivity.slack_increment_pp
    sensitivity["marginal_mean_km_per_pp"   ] = sensitivity.marginal_total_km_per_pp / n_admissions
    sensitivity["gain_km_per_added_capacity"] = sensitivity.absolute_gain_km         / sensitivity.capacity_increment.replace(0, np.nan)

    sensitivity["cumulative_gain_km" ] =       sensitivity.total_distance_km.iloc[0] - sensitivity.total_distance_km
    sensitivity["cumulative_gain_pct"] = 100 * sensitivity.cumulative_gain_km / sensitivity.total_distance_km.iloc[0]

    sensitivity["remaining_gain_to_100_km"] = sensitivity.total_distance_km - sensitivity.total_distance_km.iloc[-1]

    elbow_index, _scores = find_elbow(
        sensitivity.slack            ,
        sensitivity.total_distance_km,
    )

    elbow = None if elbow_index is None else sensitivity.loc[elbow_index]
    return elbow, elbow_index, sensitivity


@app.cell
def _(flow_metrics, nearest, observed, observed_total, pd, sensitivity):
    comparison = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "scenario" : "Observed",

                        **flow_metrics(observed, observed_total)
                    }
                ]
            ),

            sensitivity,

            pd.DataFrame(
                [
                    {
                        "scenario" : "Unconstrained",

                        **flow_metrics(nearest, observed_total)
                    }
                ]
            ),
        ],

        ignore_index=True
    )
    return (comparison,)


@app.cell(hide_code=True)
def _(comparison, mo):
    mo.ui.table(
        comparison.round(3),

        page_size=15                                        ,
        label    ="All scenarios · original route distances",
    )
    return


@app.cell(hide_code=True)
def _(elbow, elbow_index, mo, sensitivity):
    _indices = [0, len(sensitivity) - 1]

    if elbow is not None:
        _indices.insert(1, elbow_index)

    _message = (
        "No descriptive elbow was found."
        if   elbow is None
        else f"Descriptive elbow: **{elbow.slack_pct:.0f}% slack**."
    )

    mo.vstack([
        mo.md(_message),

        mo.ui.table(
            sensitivity.loc[
                _indices,

                [
                    "slack_pct"          ,
                    "total_capacity"     ,
                    "mean_distance_km"   ,
                    "cumulative_gain_pct",
                ]
            ].round(3),

            selection=None,
        ),
    ])
    return


@app.cell
def _(
    elbow,
    flow_metrics,
    n_admissions,
    nearest,
    observed_total,
    plt,
    sensitivity,
):
    mean_distance_fig, _ax = plt.subplots(figsize=(10, 4.5), layout="constrained")

    _ax.axhline(
        observed_total / n_admissions,

        color    ="#79868F"        ,
        linestyle="--"             ,
        label    ="Fluxo observado",
    )

    _ax.plot(
        sensitivity.slack_pct       ,
        sensitivity.mean_distance_km,

        "o-",

        markersize=4,
        color     ="#0072B2"            ,
        label     ="Capacidade limitada",
    )

    _ax.axhline(
        flow_metrics(
            nearest       ,
            observed_total,
        )["mean_distance_km"],

        linestyle=":"      ,
        color    ="#D55E00",
        label    ="Sem limite de capacidade",
    )

    if elbow is not None:
        _ax.axvline(
            elbow.slack_pct,

            linestyle="--"     ,
            color    ="#CC79A7",
            label    =f"Cotovelo: {elbow.slack_pct:.0f}%",
        )

    _ax.set(
        xlabel="Expansão nominal da capacidade (%)"      ,
        ylabel="Distância média (km/internação)"         ,
        title ="Expansão da capacidade e distância média",

        xticks=range(0, 101, 10),
        xlim  =(0, 100)         ,
    )

    _ax.grid(axis="y", color="#E5EAF0")

    _ax.set_axisbelow(True)

    _ax.spines[
        [
            "top"  ,
            "right",
        ]
    ].set_visible(False)

    _ax.legend(frameon=False, fontsize=9)

    mean_distance_fig
    return (mean_distance_fig,)


@app.cell
def _(elbow, n_admissions, plt, sensitivity):
    marginal_gain_fig, _ax = plt.subplots(figsize=(10, 4.5), layout="constrained")

    _ax.bar(
        sensitivity.slack_pct              .iloc[ :-1],
        sensitivity.marginal_mean_km_per_pp.iloc[1:  ],

        width=sensitivity.slack_increment_pp.iloc[1:],

        linewidth=0.5,
        align    ="edge"   ,
        color    ="#0072B2",
        edgecolor="white"  ,
    )

    if elbow is not None:
        _ax.axvline(
            elbow.slack_pct,

            color    ="#CC79A7",
            linestyle="--"     ,
            label    =f"Cotovelo: {elbow.slack_pct:.0f}%",
        )

        _ax.hlines(
            [
                elbow.cumulative_gain_km       / n_admissions / elbow.slack_pct        ,
                elbow.remaining_gain_to_100_km / n_admissions / (100 - elbow.slack_pct),
            ],

            [0              , elbow.slack_pct],
            [elbow.slack_pct, 100            ],

            color    ="#222222",
            linestyle="--"     ,
            label    ="Ganho médio por p.p.",
        )

        _ax.legend(frameon=False, fontsize=9)

    _ax.set(
        xlabel="Expansão nominal da capacidade (%)"      ,
        ylabel="Ganho no intervalo (km/internação/p.p.)" ,
        title ="Ganho marginal da expansão de capacidade",

        xticks=range(0, 101, 10),
        xlim  =(0, 100)         ,
    )

    _ax.grid(axis="y", color="#E5EAF0")

    _ax.set_axisbelow(True)

    _ax.spines[["top", "right"]].set_visible(False)

    marginal_gain_fig
    return (marginal_gain_fig,)


@app.cell
def _(
    PERIOD,
    ROOT,
    SLACK_PCT,
    comparison,
    data,
    datetime,
    flows,
    json,
    marginal_gain_fig,
    mean_distance_fig,
    nearest_path,
    od_path,
    sha256,
    timezone,
):
    _prefix = "scenario_2_capacity_constrained"

    output_paths = {
        "flows"      : ROOT / f"results/{_prefix}_{PERIOD}.csv",
        "comparison" : ROOT / f"metrics/{_prefix}_comparison_{PERIOD}.csv",
        "metadata"   : ROOT / f"metrics/scenario_2_metrics_{PERIOD}.json" ,

        "mean_distance_figure" : ROOT / f"docs/capacity_expansion_mean_distance_{PERIOD}.pdf",
        "marginal_gain_figure" : ROOT / f"docs/capacity_expansion_elbow_{PERIOD}.pdf"        ,
    }

    for _path in output_paths.values():
        _path.parent.mkdir(parents=True, exist_ok=True)


    flows.drop(
        columns=["model_distance_km"]
    ).to_csv(
        output_paths["flows"],

        index   =False      ,
        encoding="utf-8-sig",
    )

    _comparison_csv = comparison.drop(
        columns=[
            "n_admissions"                ,
            "max_interstate_admissions"   ,
            "interstate_constraint_active",
        ]
    ).astype({
        "n_saturated_providers" : "Int64",
        "interstate_admissions" : "Int64",
        "capacity_increment"    : "Int64",
    })

    _comparison_csv.to_csv(
        output_paths["comparison"],

        index       =False,
        encoding    ="utf-8-sig",
        float_format="%.2f"     ,
    )

    for _key, _figure in (
        ("mean_distance_figure", mean_distance_fig),
        ("marginal_gain_figure", marginal_gain_fig),
    ):
        _figure.savefig(
            output_paths[_key],

            bbox_inches="tight",
            metadata   ={
                "CreationDate" : None,
                "ModDate"      : None,
            },
        )

    _inputs = {f"data/{name}": digest for name, digest in data.source_hashes.items()}

    _inputs.update(
        {
            str(path.relative_to(ROOT)) : sha256(path.read_bytes()).hexdigest()
            for path in (od_path, nearest_path)
    })

    _metadata = {
        "scenario"         : 2,
        "generated_at_utc" : datetime.now(timezone.utc).isoformat(timespec="seconds"),

        "slacks"  : [pct / 100 for pct in SLACK_PCT],
        "outputs" : {
            str(path.relative_to(ROOT)) : sha256(path.read_bytes()).hexdigest()
            for key, path in output_paths.items() if key != "metadata"
        },
    }

    output_paths["metadata"].write_text(
        json.dumps(
            _metadata,

            indent      =2,
            ensure_ascii=False,
            allow_nan   =False,
        ) + "\n",

        encoding="utf-8",
    )
    return (output_paths,)


@app.cell(hide_code=True)
def _(ROOT, mo, output_paths):
    mo.md("Generated artifacts:\n\n" + "\n".join(
        f"- `{path.relative_to(ROOT)}`" for path in output_paths.values()
    ))
    return


if __name__ == "__main__":
    app.run()
