import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CAPACITY-CONSTRAINED REALLOCATION WITH INTERSTATE LIMIT
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Scenario 3 · Oncological prostatectomy · Brazil · January–June 2026.**

    Extend Scenario 2 with a national budget for admissions outside the state of residence. Keep the observed demand, provider municipalities and capacity rule. Observed production is a capacity proxy; admissions are counts, not unique patients.
    """)
    return


@app.cell
def _():
    import sys
    import json

    from pathlib  import Path
    from hashlib  import sha256
    from datetime import datetime, timezone

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
    from optimization.data      import load_instance
    from optimization.scenarios import TransportScenario

    from optimization.models.transport import build_transport_model

    from optimization.metrics    import flow_metrics
    from optimization.validation import validate_solution
    from optimization.solve      import extract_solution , solve_model, solver_options

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

    SLACKS                = (0.00 , 0.05 , 0.10 , 0.15 , 0.20 )
    MAX_INTERSTATE_SHARES = (0.000, 0.005, 0.010, 0.015, 0.020, 0.030)

    OBJECTIVE_ATOL = 1e-5
    return MAX_INTERSTATE_SHARES, OBJECTIVE_ATOL, PERIOD, SLACKS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MODEL AND REFERENCES
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For demand $D_i$, observed production $U_j^{obs}$, capacity slack $s$, interstate share $q$ and $N=\sum_iD_i$:

    $$
    \begin{aligned}
    \min_x\quad &\sum_{i,j}\tilde d_{ij}x_{ij}\\
    \text{subject to}\quad
    &\sum_jx_{ij}=D_i,\qquad\\
    &\sum_ix_{ij}\leq\lfloor(1+s)U_j^{obs}\rfloor,\\
    &\sum_{(i,j):\,state(i)\ne state(j)}x_{ij}\leq\lfloor qN\rfloor,\\
    &\qquad x_{ij}\in\mathbb Z_{\geq0}.
    \end{aligned}
    $$

    > **Notes**.: HiGHS solves the integer model using the compact instance's **rounded integer kilometres** $\tilde d_{ij}$. Reported distances use the original OD values.
    """)
    return


@app.cell
def _(
    PERIOD,
    ROOT,
    SLACKS,
    TransportScenario,
    load_instance,
    pd,
    validate_solution,
):
    data = load_instance(ROOT / "data", PERIOD)

    input_paths = {
        "od" : ROOT / f"data/origin_destination_data_{PERIOD}.csv",

        "nearest"    : ROOT / f"results/scenario_1_{PERIOD}.csv"                                ,
        "scenario_2" : ROOT / f"metrics/scenario_2_capacity_constrained_comparison_{PERIOD}.csv",
        "metadata_2" : ROOT / f"metrics/scenario_2_metrics_{PERIOD}.json"                       ,
    }

    _codes = {
        "origin_code"      : "string",
        "destination_code" : "string",
    }
    _keys = list(_codes)

    od = pd.read_csv(
        input_paths["od"],

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
            "observed_flow" : "allocated_flow",
        }
    ).copy()

    observed["same_municipality"] = observed.origin_code.eq(observed.destination_code)
    observed["total_distance_km"] = observed.allocated_flow * observed.distance_km

    nearest = pd.read_csv(input_paths["nearest"], dtype=_codes)

    observed_total    = float(observed.total_distance_km.sum   ())
    n_admissions      = sum  (data    .demand           .values())

    nearest_objective = sum  (
        data.demand[i] * min(
            data.distance[i, j]
            for j in data.destinations_by_origin[i]
        )
        for i in data.origins
    )
    observed_objective = sum(
        data.distance[row.origin_code, row.destination_code] * row.allocated_flow
        for row in observed.itertuples(index=False)
    )

    validate_solution(observed, data, TransportScenario(), observed_objective)
    validate_solution(nearest , data, TransportScenario(), nearest_objective , check_capacity=False)

    _s2 = pd.read_csv(input_paths["scenario_2"])

    s2_reference = (
        _s2
        .loc[_s2.slack.notna()]
        .set_index("slack", verify_integrity=True)
        .loc[list(SLACKS)     ]
    )
    return (
        data,
        distances,
        n_admissions,
        nearest,
        nearest_objective,
        observed,
        observed_objective,
        observed_total,
        s2_reference,
    )


@app.cell
def _(SLACKS, TransportScenario, data, pd):
    _state_demand = (
        pd
        .Series (data.demand                   )
        .groupby(pd  .Series(data.origin_state))
        .sum    ()
    )

    minimum_interstate = {}

    for _slack in SLACKS:
        _scenario = TransportScenario(slack=_slack)

        _capacity = pd.Series(_scenario.capacities(data.observed_supply))

        _state_capacity = (
            _capacity
            .groupby(pd.Series(data.provider_state)   )
            .sum    ()
            .reindex(_state_demand.index, fill_value=0)
        )

        minimum_interstate[_slack] = int(
            (_state_demand - _state_capacity).clip(lower=0).sum()
        )
    return (minimum_interstate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CAPACITY AND INTERSTATE GRID
    """)
    return


@app.cell
def _(
    MAX_INTERSTATE_SHARES,
    SLACKS,
    TransportScenario,
    build_transport_model,
    data,
    distances,
    extract_solution,
    flow_metrics,
    minimum_interstate,
    mo,
    n_admissions,
    np,
    observed,
    observed_total,
    pd,
    pyo,
    s2_reference,
    solve_model,
    validate_solution,
):
    _grid = [
        (s, q)
        for s in SLACKS
        for q in MAX_INTERSTATE_SHARES
    ]

    _metric_names = [
        *flow_metrics(observed, observed_total),

        "model_objective_km"                          ,
        "objective_error_km"                          ,
        "n_saturated_providers"                       ,
        "interstate_admissions"                       ,
        "interstate_constraint_active"                ,
        "capacity_utilization_pct"                    ,
        "model_distance_penalty_vs_scenario_2_km"     ,
        "model_mean_distance_penalty_vs_scenario_2_km",
    ]

    _rows, _solutions = [], []

    for _slack, _share in mo.status.progress_bar(_grid, title="Solving interstate scenarios"):
        _scenario = TransportScenario(slack=_slack, max_interstate_share=_share)

        _model  = build_transport_model(data  , _scenario)
        _result = solve_model          (_model, allow_infeasible=True)

        _termination    = str(_result  .solver    .termination_condition         )
        _total_capacity = sum(_scenario.capacities(data.observed_supply).values())

        _row = {
            "slack"                     : _slack,
            "max_interstate_share"      : _share,
            "max_interstate_admissions" : _scenario.interstate_limit(n_admissions),

            "solver_status"         : str(_result.solver.status),
            "termination_condition" : _termination              ,
            "feasible"              : _termination == "optimal" ,

            "total_capacity"              : _total_capacity           ,
            "minimum_interstate_required" : minimum_interstate[_slack],

            **dict.fromkeys(_metric_names, np.nan),
        }

        if _row["feasible"]:
            _objective = float(pyo.value(_model.total_distance))

            _solution = extract_solution (_model, data, distances=distances)
            _checks   = validate_solution(
                _solution ,
                data      ,
                _scenario ,
                _objective,

                termination_condition=_termination         ,
                solver_status        =_row["solver_status"],
            )

            _penalty = _objective - s2_reference.loc[_slack, "model_objective_km"]

            _row.update({
                **flow_metrics(_solution, observed_total),

                **_checks,

                "model_objective_km"       : _objective                          ,
                "capacity_utilization_pct" : 100 * n_admissions / _total_capacity,

                "model_distance_penalty_vs_scenario_2_km"      : _penalty               ,
                "model_mean_distance_penalty_vs_scenario_2_km" : _penalty / n_admissions,
            })

            _solutions.append(
                _solution.assign(
                    slack               =_slack,
                    max_interstate_share=_share,
                )
            )

        _rows.append(_row)

        del _model, _result


    comparison = pd.DataFrame(_rows)

    comparison["interstate_constraint_active"] = comparison.interstate_constraint_active.astype("boolean")

    flows = (
        pd.concat(_solutions, ignore_index=True)
        if   _solutions
        else pd.DataFrame(
            columns=[
                "origin_code"         ,
                "destination_code"    ,
                "allocated_flow"      ,
                "model_distance_km"   ,
                "distance_km"         ,
                "total_distance_km"   ,
                "same_municipality"   ,
                "same_state"          ,
                "slack"               ,
                "max_interstate_share",
            ]
        )
    )

    flows = flows[
        [
            "slack"               ,
            "max_interstate_share",

            *[
                name
                for name in flows.columns
                if  name not in (
                    "slack"               ,
                    "max_interstate_share",
                )
            ]
        ]
    ]
    return comparison, flows


@app.cell
def _(
    OBJECTIVE_ATOL,
    comparison,
    nearest_objective,
    observed,
    observed_objective,
    s2_reference,
):
    if not comparison.feasible.eq(
        comparison.max_interstate_admissions.ge(comparison.minimum_interstate_required)
    ).all():
        raise RuntimeError("Solver feasibility differs from the state-capacity balance.")

    _observed_interstate = int(observed.loc[~observed.same_state, "allocated_flow"].sum())

    for _row in comparison.loc[comparison.feasible].itertuples(index=False):
        _relaxed = comparison.loc[
            comparison.slack                    .ge(_row.slack                    ) &
            comparison.max_interstate_admissions.ge(_row.max_interstate_admissions)
        ]
        if not (
            _relaxed.feasible                                                        .all() and
            _relaxed.model_objective_km.le (_row.model_objective_km + OBJECTIVE_ATOL).all()
        ):
            raise RuntimeError("Feasibility or the integer objective worsened under relaxation.")

        _s2 = s2_reference.loc[_row.slack]

        if _row.model_objective_km < max(nearest_objective, _s2.model_objective_km) - OBJECTIVE_ATOL:
            raise RuntimeError("Integer objective is below a reference lower bound.")

        if (
                _s2 .n_outside_state                              <= _row.max_interstate_admissions and
            abs(_row.model_objective_km - _s2.model_objective_km) >  OBJECTIVE_ATOL
        ):
            raise RuntimeError("An admissible Scenario 2 optimum was not recovered.")

        if (
            _observed_interstate    <= _row.max_interstate_admissions and
            _row.model_objective_km >  observed_objective + OBJECTIVE_ATOL
        ):
            raise RuntimeError("Integer objective exceeds the admissible observed allocation.")

    grid_summary = (
        f"{int(comparison.feasible.sum())} optimal · {int((~comparison.feasible).sum())} infeasible · grid checks passed."
    )
    return (grid_summary,)


@app.cell(hide_code=True)
def _(
    comparison,
    flow_metrics,
    grid_summary,
    mo,
    nearest,
    observed,
    observed_total,
    pd,
    s2_reference,
):
    _references = pd.DataFrame([
        {
            "scenario" : name,

            **flow_metrics(table, observed_total),
        }
        for name, table in (
            ("Observed"     , observed),
            ("Unconstrained", nearest ),
        )
    ])

    mo.vstack([
        mo.md(grid_summary),

        mo.ui.table(
            pd.concat(
                [
                    _references,
                    s2_reference.reset_index().assign(scenario="Scenario 2"),
                ],

                ignore_index=True,
            ).round(3),

            selection=None,
            label    ="Reference allocations",
        ),

        mo.ui.table(
            comparison.round(
                {
                    name : 4
                    if   name == "max_interstate_share"
                    else 3
                    for name in comparison.columns
                }
            ),

            page_size=30,
            label    ="All 30 combinations · original route distances",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DISTANCES AND INTERSTATE ADMISSIONS
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Total distance weights each route by admissions; mean distance divides by all admissions. Quantiles use distances repeated by integer admission frequency. Grey heatmap cells indicate infeasible combinations.
    """)
    return


@app.cell
def _(MAX_INTERSTATE_SHARES, SLACKS, comparison, mo, np, plt):
    heatmap_figures = {}

    for _metric, _title, _unit, _palette, _name in (
        ("mean_distance_km"      , "Distância média"                  , "km/internação"   , "viridis_r", "mean_distance"     ),
        ("max_distance_km"       , "Distância máxima"                 , "km"              , "viridis_r", "max_distance"      ),
        ("pct_outside_state"     , "Internações entre estados"        , "% de internações", "viridis"  , "interstate_share"  ),
        ("distance_reduction_pct", "Redução de distância vs observado", "%"               , "viridis"  , "distance_reduction"),
    ):
        _values = comparison.pivot(
            index  ="max_interstate_share",
            columns="slack"               ,
            values =_metric,
        ).reindex(
            index  =MAX_INTERSTATE_SHARES,
            columns=SLACKS               ,
        ).to_numpy(dtype=float)

        _fig, _ax = plt.subplots(figsize=(6, 4), layout="constrained")

        _cmap = plt.get_cmap(_palette).copy()
        _cmap.set_bad("#E5E7EB")

        _image = _ax.imshow(np.ma.masked_invalid(_values), cmap=_cmap, aspect="auto")

        _ax.set_xticks(range(len(SLACKS               )), [f"{100*s:g}"   for s in SLACKS               ])
        _ax.set_yticks(range(len(MAX_INTERSTATE_SHARES)), [f"{100*q:.2f}" for q in MAX_INTERSTATE_SHARES])

        _ax.set(
            xlabel="Expansão nominal da capacidade (%)",
            ylabel="Limite interestadual (%)"          ,

            title=_title,
        )

        _ax          .tick_params(length=0)
        _ax.spines[:].set_visible(False   )

        for (_i, _j), _value in np.ndenumerate(_values):
            _rgba = _image.cmap(_image.norm(_value)) if np.isfinite(_value) else (1, 1, 1, 1)

            _ax.text(
                _j,
                _i,

                "Inviável" if np.isnan(_value) else f"{_value:.2f}",

                fontsize=9,
                ha      ="center",
                va      ="center",
                color   ="white" if np.dot(_rgba[:3], [0.2126, 0.7152, 0.0722]) < 0.45 else "#172534",
            )

        _fig.colorbar(_image, ax=_ax, label=_unit, shrink=0.86)

        heatmap_figures[f"scenario_3_{_name}_heatmap"] = _fig

    mo.vstack(list(heatmap_figures.values()))
    return (heatmap_figures,)


@app.cell
def _(
    SLACKS,
    comparison,
    flow_metrics,
    n_admissions,
    nearest,
    observed_total,
    plt,
    s2_reference,
):
    tradeoff_fig, _ax = plt.subplots(figsize=(10, 4.5), layout="constrained")

    _ax.axhline(
        observed_total / n_admissions,

        linestyle="-."             ,
        color    ="#79868F"        ,
        label    ="Fluxo observado",
    )

    for _slack, _color in zip(SLACKS, ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6A51A3")):
        _subset = comparison.loc[comparison.slack.eq(_slack)]

        _ax.plot(
            100 * _subset.max_interstate_share,
                  _subset.mean_distance_km    ,

            "o-",

            markersize=4     ,
            color     =_color,
            label     =f"Cenário 3 · Folga {_slack:.0%}",
        )

        _ax.axhline(
            s2_reference.loc[_slack, "mean_distance_km"],

            alpha    =0.7   ,
            linewidth=1     ,
            linestyle="--"  ,
            color    =_color,
        )

    _ax.plot(
        [],
        [],

        linestyle="--"     ,
        color    ="#79868F",
        label    ="Cenário 2 · mesma folga",
    )

    _ax.axhline(
        flow_metrics(
            nearest       ,
            observed_total,
        )["mean_distance_km"],

        linestyle=":"      ,
        color    ="#172534",
        label    ="Sem limite de capacidade",
    )

    _ax.set(
        xlabel="Limite interestadual (%)"                 ,
        ylabel="Distância média (km/internação)"          ,
        title ="Restrição interestadual e distância média",

        xticks=[0, 0.5, 1, 2, 3],
    )

    _ax.grid         (axis="y", color="#E5EAF0")
    _ax.set_axisbelow(True)

    _ax.spines[["top", "right"]].set_visible(False)

    _ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)

    tradeoff_fig
    return (tradeoff_fig,)


@app.cell(hide_code=True)
def _(MAX_INTERSTATE_SHARES, SLACKS, comparison, minimum_interstate, mo, pd):
    _means = comparison.pivot(
        index  ="max_interstate_share",
        columns="slack"               ,
        values ="mean_distance_km"    ,
    ).reindex(
        index  =MAX_INTERSTATE_SHARES,
        columns=SLACKS               ,
    )

    _gains = pd.DataFrame({
        f"{left:.0%} → {right:.0%}": _means[left] - _means[right]
        for left, right in zip(SLACKS[:-1], SLACKS[1:])
    })

    _frontier = comparison.loc[comparison.feasible].groupby("slack").agg(
        minimum_feasible_max_interstate_share     =("max_interstate_share"     , "min"),
        minimum_feasible_max_interstate_admissions=("max_interstate_admissions", "min"),
    ).reindex(SLACKS)

    _frontier["minimum_interstate_required"] = pd.Series(minimum_interstate)

    _active = comparison.pivot(
        index  ="max_interstate_share"        ,
        columns="slack"                       ,
        values ="interstate_constraint_active",
    )

    mo.vstack([
        mo.md("## MARGINAL GAINS AND FEASIBILITY\n\nThe frontier reports the first feasible budget **in this grid**."),

        mo.ui.table(_gains   .round(3).reset_index(), selection=None, label="Marginal mean-distance gains (km/internação)"),
        mo.ui.table(_frontier         .reset_index(), selection=None, label="Feasibility frontier"                        ),
        mo.ui.table(_active           .reset_index(), selection=None, label="Interstate constraint active"                ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MINIMUM EXPANSION FOR ZERO-INTERSTATE FEASIBILITY
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To find the smallest **uniform capacity slack**, fix $q=0$ and retain only within-state pairs $A_0=\{(i,j):state(i)=state(j)\}$:

    $$
    \begin{aligned}
    \min_{s,x}\quad &s\\
    \text{subject to}\quad
    &\sum_{j:(i,j)\in A_0}x_{ij}=D_i,\\
    &\sum_{i:(i,j)\in A_0}x_{ij}\le(1+s)U_j^{obs},\\
    &x_{ij}\in\mathbb Z_{\ge0},\\
    &s\ge0.
    \end{aligned}
    $$

    With complete within-state connectivity, feasibility is equivalent to each state's expanded capacity covering its demand. The exact threshold here is $s^*=1/12=8\frac13\%$: Pará's production of $(14,6,12)$ becomes capacity $(15,6,13)$, meeting demand of $34$. All other states are feasible at this slack.

    > **Note 01**.: This is a mixed-integer linear model: $U_j^{obs}$ is constant, and integer allocations make the capacity bound equivalent to $\lfloor(1+s)U_j^{obs}\rfloor$.

    > **Note 02**.: Distance can be minimized afterwards with $s=s^*$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GENERATE ARTIFACTS
    """)
    return


@app.cell
def _(
    MAX_INTERSTATE_SHARES,
    PERIOD,
    ROOT,
    SLACKS,
    comparison,
    flows,
    heatmap_figures,
    tradeoff_fig,
):
    _prefix  = "scenario_3_interstate_constrained"

    _figures = {
        **heatmap_figures,

        "scenario_3_tradeoff" : tradeoff_fig,
    }

    output_paths = {
        "flows"      : ROOT / f"results/{_prefix}_{PERIOD}.csv"           ,
        "comparison" : ROOT / f"metrics/{_prefix}_comparison_{PERIOD}.csv",

        **{
            name : ROOT / f"docs/{name}_{PERIOD}.pdf"
            for name in _figures
        },
    }

    for _path in output_paths.values():
        _path.parent.mkdir(parents=True, exist_ok=True)

    flows.drop(
        columns=[
            "slack_pct"        ,
            "alpha"            ,
            "model_distance_km",
            "same_municipality",
        ],

        errors="ignore",
    ).to_csv(
        output_paths["flows"],

        index   =False      ,
        encoding="utf-8-sig",
    )

    _counts = [
        "n_admissions"          ,
        "n_active_od_pairs"     ,
        "n_saturated_providers" ,
        "interstate_admissions" ,
        "n_same_municipality"   ,
        "n_outside_municipality",
        "n_same_state"          ,
        "n_outside_state"       ,
    ]

    _comparison_csv = comparison.astype(dict.fromkeys(_counts, "Int64"))

    _comparison_csv["max_interstate_share"] = comparison.max_interstate_share.map("{:.3f}".format)

    _comparison_csv.drop  (
        columns=[
            "solver_status",
            "feasible"     ,
            "n_admissions" ,
        ],

        inplace=True,
    )
    _comparison_csv.drop(
        index  =_comparison_csv.index[
            ~comparison["feasible"]
        ],

        inplace=True,
    )

    _comparison_csv.to_csv(
        output_paths["comparison"],

        index       =False,
        encoding    ="utf-8-sig",
        float_format="%.2f"     ,
    )

    for _name, _figure in _figures.items():
        _figure.savefig(
            output_paths[_name],

            bbox_inches="tight",
            metadata   ={
                "CreationDate" : None,
                "ModDate"      : None,
            },
        )

    grid_metadata = {
        "slacks"                : list(SLACKS               ),
        "max_interstate_shares" : list(MAX_INTERSTATE_SHARES),
        "n_feasible"            : int(  comparison.feasible .sum()),
        "n_infeasible"          : int((~comparison.feasible).sum()),

        "outputs" : [
            str(path.relative_to(ROOT))
            for path in output_paths.values()
        ],
    }
    return grid_metadata, output_paths


@app.cell(hide_code=True)
def _(ROOT, mo, output_paths):
    mo.md("Generated artifacts:\n\n" + "\n".join(
        f"- `{path.relative_to(ROOT)}`"
        for path in output_paths.values()
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RELATIVE COUNTERFACTUALS: MINIMUM INTERSTATE SHARE
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For each slack $s$, let $F_s$ contain all nonnegative **integer** allocations satisfying demand and provider capacities $\lfloor(1+s)U_j^{obs}\rfloor$. Define rounded travel distance $C(x)=\sum_{ij}\tilde d_{ij}x_{ij}$ and interstate admissions $K(x)=\sum_{(i,j):state(i)\ne state(j)}x_{ij}$.

    First solve the unrestricted reference, separately at each expansion:

    $$Z_s^*=\min_{x\in F_s}C(x),\qquad q=1.$$

    Then find the smallest interstate budget satisfying a relative distance target against that **fixed reference value**:

    $$
    \begin{aligned}
    \min_{x,q}\quad &q\\
    \text{subject to}\quad
    &C(x)\le\alpha Z_s^*,\\
    &K(x)\le qN,\\
    &N=\sum_iD_i,\\
    &0\le q\le1,\\
    &x\in F_s.
    \end{aligned}
    $$

    Since $N$ is fixed, solve for the smallest integer $K(x)$ and report $q_{min}=K_{min}/N$ exactly. For **$\alpha=1$**, every admissible allocation has distance $Z_s^*$: this selects the fewest interstate admissions among distance-optimal solutions.

    > **Note**.: The objective is the smallest admissible $q$, $q=1$ is the reference's allowed ceiling, not its realized share. The minimum share can increase with expansion because $Z_s^*$ also changes.
    """)
    return


@app.cell
def _():
    CF_SLACKS = tuple(
        pct / 100
        for pct in (*range(26), *range(30, 101, 5))
    )

    CF_ALPHA  = 1
    return CF_ALPHA, CF_SLACKS


@app.cell
def _(
    CF_ALPHA,
    CF_SLACKS,
    OBJECTIVE_ATOL,
    TransportScenario,
    build_transport_model,
    data,
    distances,
    extract_solution,
    mo,
    n_admissions,
    np,
    pd,
    pyo,
    solver_options,
    validate_solution,
):
    _rows, _allocations = [], []

    for _slack in mo.status.progress_bar(CF_SLACKS, title="Solving relative counterfactuals"):
        _scenario = TransportScenario    (
            slack               =_slack,
            max_interstate_share=1     ,
        )
        _model    = build_transport_model(
            data     ,
            _scenario,
        )

        _solver = pyo.SolverFactory("appsi_highs")

        _solver.options.update(
            {
                **solver_options(_model),

                "presolve" : "off",
            }
        )

        def _solve(warmstart=False):
            _result = _solver.solve(
                _model,

                load_solutions=False    ,
                warmstart     =warmstart,
            )

            if _result.solver.termination_condition != pyo.TerminationCondition.optimal:
                raise RuntimeError(f"An optimal solution is required at slack={_slack}.")

            _model.solutions.load_from(_result)

            _cost         = float(pyo.value(_model.total_distance))
            _integer_cost = int  (round    (_cost                ))

            if not np.isclose(_cost, _integer_cost, rtol=0, atol=OBJECTIVE_ATOL):
                raise RuntimeError("The integer-distance objective is not integral.")

            _solution = extract_solution (_model, data, distances=distances)
            _checks   = validate_solution(
                _solution    ,
                data         ,
                _scenario    ,
                _integer_cost,

                termination_condition=_result.solver.termination_condition,
                solver_status        =_result.solver.status               ,
            )

            return _solution, _integer_cost, int(_checks["interstate_admissions"])

        _reference, _reference_cost, _reference_k = _solve()

        _budget = int(CF_ALPHA * _reference_cost)

        _model.relative_distance_limit = pyo.Constraint(
            expr=_model.total_distance.expr <= _budget
        )

        _model.total_distance.deactivate()

        _model.minimum_interstate      = pyo.Objective(
            expr =pyo.quicksum(
                _model.x[arc]
                for arc in _model.A_inter
            ),

            sense=pyo.minimize,
        )

        _counterfactual, _cost, _k = _solve(warmstart=True)

        if (
            _cost > _budget or

            (CF_ALPHA == 1 and _cost != _reference_cost) or
            (CF_ALPHA >= 1 and _k    >  _reference_k   ) or

            not np.isclose(
                pyo.value(_model.minimum_interstate),
                _k                                  ,

                rtol=0             ,
                atol=OBJECTIVE_ATOL,
            )
        ):
            raise RuntimeError("The relative distance or interstate objective failed validation.")

        _rows.append({
            "slack"          : _slack      ,
            "n_admissions"   : n_admissions,
            "total_capacity" : sum(_scenario.capacities(data.observed_supply).values()),

            "reference_model_objective_km"      : _reference_cost,
            "relative_distance_limit_km"        : _budget        ,
            "counterfactual_model_objective_km" : _cost          ,

            "unrestricted_interstate_admissions"   : _reference_k,
            "counterfactual_interstate_admissions" : _k          ,

            "unrestricted_interstate_pct"     : 100 * _reference_k / n_admissions,
            "counterfactual_interstate_pct"   : 100 * _k / n_admissions          ,
            "interstate_admissions_reduction" : _reference_k - _k                ,
        })

        for _stage, _solution in (
            ("unrestricted"           , _reference     ),
            ("relative_counterfactual", _counterfactual),
        ):
            _allocations.append(
                _solution
                .drop  (
                    columns=[
                        "model_distance_km",
                        "same_municipality",
                    ]
                )
                .assign(
                    stage=_stage,
                    slack=_slack,
                )
            )


    cf_comparison = pd.DataFrame(_rows)

    cf_flows = pd.concat(_allocations, ignore_index=True)
    cf_flows = cf_flows [
        [
            "stage",
            "slack",

            *[
                c
                for c in cf_flows
                if  c not in {"stage", "slack"}
            ],
        ]
    ]

    if cf_comparison.reference_model_objective_km.diff().dropna().gt(OBJECTIVE_ATOL).any():
        raise RuntimeError("The optimal reference distance increased with capacity.")
    return cf_comparison, cf_flows


@app.cell(hide_code=True)
def _(cf_comparison, mo):
    mo.ui.table(
        cf_comparison.round(6),

        selection=None                                   ,
        label    ="Relative counterfactuals by expansion",
    )
    return


@app.cell
def _(cf_comparison, plt):
    cf_figure, _ax = plt.subplots(figsize=(10, 4.8), layout="constrained")

    for _column, _style, _color, _label in (
        ("unrestricted_interstate_pct"  , "o-" , "#0072B2", "Unrestricted Scenario 3 (q = 1)"           ),
        ("counterfactual_interstate_pct", "s--", "#D55E00", "Relative counterfactual (α = 1, minimum q)"),
    ):
        _ax.plot(
            100 * cf_comparison.slack   ,
                  cf_comparison[_column],

            _style,

            markersize=4     ,
            color     =_color,
            label     =_label,
        )

    _ax.set(
        xlabel="Nominal capacity expansion (%)",
        ylabel="Interstate admissions (%)"     ,
        title ="Interstate share at the same optimal distance",

        xticks=range(0, 101, 10),
        xlim  =(0, 100 ),
        ylim  =(0, None),
    )

    _ax.grid(
        axis ="y"      ,
        color="#E5EAF0",
    )

    _ax.set_axisbelow(True)

    _ax.spines[["top", "right"]].set_visible(False)
    _ax.legend(frameon=False, fontsize=9)

    cf_figure
    return (cf_figure,)


@app.cell
def _(CF_ALPHA, CF_SLACKS, PERIOD, ROOT, cf_comparison, cf_figure, cf_flows):
    _prefix = "scenario_3_relative_counterfactual"

    cf_output_paths = {
        "unrestricted_flows" : ROOT / f"results/scenario_3_unrestricted_{PERIOD}.csv",
        "flows"              : ROOT / f"results/{_prefix}_{PERIOD}.csv"              ,
        "comparison"         : ROOT / f"metrics/{_prefix}_comparison_{PERIOD}.csv"   ,
        "figure"             : ROOT / f"docs/{_prefix}_interstate_share_{PERIOD}.pdf",
    }

    for _path in cf_output_paths.values():
        _path.parent.mkdir(parents=True, exist_ok=True)

    for _key, _stage in (
        ("unrestricted_flows", "unrestricted"           ),
        ("flows"             , "relative_counterfactual"),
    ):
        cf_flows.loc[
            cf_flows.stage.eq(_stage)
        ].drop(
            columns="stage",
        ).to_csv(
            cf_output_paths[_key],

            index   =False      ,
            encoding="utf-8-sig",
        )

    cf_comparison.to_csv(
        cf_output_paths["comparison"], index=False, encoding="utf-8-sig"
    )

    cf_figure.savefig(
        cf_output_paths["figure"],

        bbox_inches="tight",
        metadata   ={
            "CreationDate" : None,
            "ModDate"      : None,
        }
    )

    cf_metadata = {
        "alpha"  :      CF_ALPHA  ,
        "slacks" : list(CF_SLACKS),

        "n_reference_optima"      : len(cf_comparison),
        "n_counterfactual_optima" : len(cf_comparison),

        "outputs" : [
            str(path.relative_to(ROOT))
            for path in cf_output_paths.values()
        ],
    }
    return cf_metadata, cf_output_paths


@app.cell
def _(
    PERIOD,
    ROOT,
    cf_metadata,
    datetime,
    grid_metadata,
    json,
    sha256,
    timezone,
):
    metadata_path = ROOT / f"metrics/scenario_3_metrics_{PERIOD}.json"

    _metadata = {
        "scenario"         : 3,
        "generated_at_utc" : datetime.now(timezone.utc).isoformat(timespec="seconds"),

        "interstate_grid"         : {k: v for k, v in grid_metadata.items() if k not in {"outputs"}},
        "relative_counterfactual" : {k: v for k, v in cf_metadata  .items() if k not in {"outputs"}},

        **{
            group : {
                name : sha256((ROOT / name).read_bytes()).hexdigest()
                for name in sorted(set(grid_metadata[group]) | set(cf_metadata[group]))
            }

            for group in ("outputs",)
        },
    }

    metadata_path.write_text(
        json.dumps(
            _metadata,

            indent      =2    ,
            ensure_ascii=False,
            allow_nan   =False,
        ) + "\n",

        encoding="utf-8",
    )
    return (metadata_path,)


@app.cell(hide_code=True)
def _(ROOT, cf_output_paths, metadata_path, mo):
    mo.md("Generated counterfactual artifacts:\n\n" + "\n".join(
        f"- `{path.relative_to(ROOT)}`" for path in (*cf_output_paths.values(), metadata_path)
    ))
    return


if __name__ == "__main__":
    app.run()
