import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TERRITORIAL EQUITY UNDER CAP25
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Oncological prostatectomy · Brazil · January–June 2026.**

    Three experiments:

    1. Distribution of gains/losses;
    2. Protection of municipal means;
    3. An efficiency–equity frontier.

    > **Note**.: CAP25 means capacity $\lfloor1.25U_j^{obs}\rfloor$ at each provider municipality.
    """)
    return


@app.cell
def _():
    import sys
    import json

    from pathlib            import Path
    from hashlib            import sha256
    from importlib.metadata import version
    from io                 import StringIO

    import numpy             as np
    import pandas            as pd
    import marimo            as mo
    import matplotlib.pyplot as plt

    return Path, StringIO, json, mo, np, pd, sha256, sys, version


@app.cell
def _(Path, sys):
    ROOT = next(
        p
        for p in (
             Path.cwd()        ,
            *Path.cwd().parents,
        )
        if (p / "pyproject.toml").is_file()
    )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return (ROOT,)


@app.cell
def _():
    from optimization.data       import build_instance
    from optimization.validation import validate_solution
    from optimization.scenarios  import TransportScenario

    from optimization.equity import compare_origins  , \
                                    distance_budget  , \
                                    origin_means     , \
                                    scenario_metrics , \
                                    solve_equity     , \
                                    summarize_origins

    return (
        TransportScenario,
        build_instance,
        compare_origins,
        distance_budget,
        scenario_metrics,
        solve_equity,
        summarize_origins,
        validate_solution,
    )


@app.cell
def _():
    PERIOD = "2026_first_half"

    SLACK = 0.25

    EPSILONS         = (0  , 0.01, 0.02, 0.05, 0.10)
    THRESHOLDS_KM    = (100, 200 , 500 , 850 , 1000)
    PROXIMITY_SHARES = (0.5, 0.9 , 1.0 )
    return EPSILONS, PERIOD, PROXIMITY_SHARES, SLACK, THRESHOLDS_KM


@app.cell
def _(
    PERIOD,
    ROOT,
    SLACK,
    TransportScenario,
    build_instance,
    np,
    pd,
    validate_solution,
):
    input_paths = [
        ROOT / f"data/origin_destination_data_{PERIOD}.csv"           ,
        ROOT / f"data/municipal_data_{PERIOD}.csv"                    ,
        ROOT / f"results/scenario_2_capacity_constrained_{PERIOD}.csv",
    ]

    _codes = {
        "origin_code"      : "string",
        "destination_code" : "string",
    }

    od        = pd.read_csv(input_paths[0], dtype=_codes                         )
    municipal = pd.read_csv(input_paths[1], dtype={"municipality_code": "string"})

    data = build_instance(od, municipal)

    distances = od.set_index(list(_codes)).distance_km

    observed = od.loc[
        od.observed_flow.gt(0)
    ].rename(
        columns={
            "observed_flow" : "allocated_flow",
        }
    ).copy()
    observed["total_distance_km"] = observed.allocated_flow * observed.distance_km

    saved_flows = pd.read_csv(input_paths[2], dtype=_codes)

    cap25 = saved_flows.loc[np.isclose(saved_flows.slack, SLACK)].drop(columns="slack").copy()
    cap0  = saved_flows.loc[saved_flows.slack.eq(0)             ].drop(columns="slack").copy()

    for _frame, _slack in (
        (observed, 0    ),
        (cap0    , 0    ),
        (cap25   , SLACK),
    ):
        _frame["same_municipality"] = _frame.origin_code.eq(_frame.destination_code)

        _arcs = list(
            zip(
                _frame.origin_code     ,
                _frame.destination_code,
            )
        )

        _frame["model_distance_km"] = [
            data.distance[a]
            for a in _arcs
        ]

        validate_solution(
            _frame,
            data  ,

            TransportScenario(slack=_slack),

            float(
                (_frame.model_distance_km * _frame.allocated_flow).sum()
            ),
        )

    reference_cost = int((cap25.model_distance_km * cap25.allocated_flow).sum())
    return (
        cap0,
        cap25,
        data,
        distances,
        input_paths,
        municipal,
        observed,
        reference_cost,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## WHO GAINS AND WHO LOSES?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For each origin

    $m_i(x)=\sum_jd_{ij}x_{ij}/D_i \quad\text{and}\quad \Delta_i=m_i(CAP25)-m_i(obs)$.

    Positive values indicate a worse municipal mean. Both equal municipality weights and admission weights are reported. Admissions in worsening origins are **not identified individual losers**.
    """)
    return


@app.cell
def _(cap25, compare_origins, mo, municipal, observed, pd, summarize_origins):
    impact = compare_origins(observed, cap25, municipal)

    distribution = pd.concat(
        [
            summarize_origins(impact),
            *[
                summarize_origins(impact, by)
                for by in (
                    "state"                 ,
                    "region"                ,
                    "observed_distance_band",
                )
            ],
        ],

        ignore_index=True,
    )

    mo.ui.table(distribution, selection=None)
    return distribution, impact


@app.cell
def _(impact, mo):
    mo.vstack([
        mo.md("Largest increases in municipal mean distance:"),

        mo.ui.table(
            impact.nlargest(15, "change_km")[
                [
                    "municipality"    ,
                    "state"           ,
                    "demand"          ,
                    "observed_mean_km",
                    "scenario_mean_km", 
                ]
            ].round(2),

            selection=None,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MUNICIPAL PROTECTION
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Minimize total travel subject to CAP25 and $\sum_jd_{ij}x_{ij}\leq\sum_jd_{ij}x_{ij}^{obs}$ for every origin. Observed flows certify feasibility. Protection applies to municipal means, and can preserve historical inequalities.
    """)
    return


@app.cell
def _(
    SLACK,
    cap0,
    cap25,
    data,
    distances,
    municipal,
    observed,
    pd,
    reference_cost,
    scenario_metrics,
    solve_equity,
):
    protected, protection_audit = solve_equity(
        data     ,
        distances,

        slack     =SLACK                                                  ,
        protection=observed.groupby("origin_code").total_distance_km.sum(),
    )

    if protected is None:
        raise RuntimeError("Observed flows should certify protection feasibility.")

    comparison = pd.DataFrame([
        scenario_metrics("Observed"       , observed , observed, municipal),
        scenario_metrics("CAP0"           , cap0     , observed, municipal),
        scenario_metrics("CAP25"          , cap25    , observed, municipal),
        scenario_metrics("CAP25 protected", protected, observed, municipal, protection_audit),
    ])

    comparison["extra_model_distance_vs_cap25_pct"] = 100 * (
        comparison.model_objective_km / reference_cost - 1
    )
    return comparison, protected


@app.cell
def _(comparison, mo):
    mo.ui.table(
        comparison[
            [
                "scenario",

                "n_active_od_pairs"    ,
                "total_distance_km"    ,
                "mean_distance_km"     ,
                "max_distance_km"      ,

                "improved_origins" ,
                "unchanged_origins",
                "worsened_origins" ,
            ]
        ],

        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EFFICIENCY–EQUITY FRONTIER
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This experiment investigates **how much the worst municipal mean travel distance can be reduced by allowing a controlled increase in total national travel distance**. For each tolerance \(\varepsilon\), a distance budget is imposed relative to the optimal cost of the original CAP model, \(C^*\):

    \[
    C(x)\leq\lfloor(1+\varepsilon)C^*\rfloor.
    \]

    Optimization proceeds in two stages:

    1. **Equity (minimax):** Minimize the largest mean travel distance across municipalities of origin:

       \[
       \min_x\max_i\left(\frac{\sum_j d_{ij}x_{ij}}{D_i}\right).
       \]

    2. **Efficiency:** Preserve the optimal minimax value (within a tolerance of \(10^{-5}\) km) while minimizing total national travel distance to avoid unnecessary additional travel.

    **Key cases:**

    - **\(\varepsilon=0\):** Examines whether alternative distance-optimal allocations provide better territorial equity without sacrificing efficiency.
    - **\(\varepsilon>0\):** Allows a controlled efficiency loss to reduce the worst municipal mean travel distance.
    - **Without municipal protection:** Some municipalities may experience increased travel distances because the objective is to minimize the worst municipal mean rather than guarantee individual municipal improvements.
    """)
    return


@app.cell
def _(
    EPSILONS,
    SLACK,
    data,
    distance_budget,
    distances,
    mo,
    municipal,
    observed,
    pd,
    reference_cost,
    scenario_metrics,
    solve_equity,
):
    _rows, frontier_allocations = [], {}
    for _epsilon in mo.status.progress_bar(EPSILONS, title="Solving equity frontier"):
        _budget = distance_budget(reference_cost, _epsilon)

        _name = f"frontier_{_epsilon:.0%}"

        _solution, _audit = solve_equity(
            data     ,
            distances,

            slack  =SLACK  ,
            budget =_budget,
            minimax=True   ,
        )

        if _solution is None:
            raise RuntimeError("CAP25 certifies every frontier budget is feasible.")

        _rows.append({
            "epsilon"   : _epsilon,
            "budget_km" : _budget ,

            **scenario_metrics(
                _name    ,
                _solution,
                observed ,
                municipal,
                _audit   ,
            ),
        })

        frontier_allocations[_name] = _solution

    frontier = pd.DataFrame(_rows)

    frontier["extra_model_distance_pct"] = 100 * (frontier.model_objective_km / reference_cost - 1)

    if frontier.max_municipal_mean_km.diff().dropna().gt(2e-5).any():
        raise RuntimeError("Equity worsened when its distance budget increased.")

    mo.ui.table(
        frontier[
            [
                "epsilon"  ,
                "budget_km",

                "total_distance_km",
                "mean_distance_km" ,
                "max_distance_km"  ,

                "improved_origins" ,
                "unchanged_origins",
                "worsened_origins" ,
            ]
        ],

        selection=None,
    )
    return frontier, frontier_allocations


@app.cell
def _(
    EPSILONS,
    PERIOD,
    PROXIMITY_SHARES,
    ROOT,
    SLACK,
    StringIO,
    THRESHOLDS_KM,
    TransportScenario,
    cap25,
    compare_origins,
    comparison,
    data,
    distribution,
    equity_figure,
    frontier,
    frontier_allocations,
    impact,
    input_paths,
    json,
    municipal,
    observed,
    pd,
    protected,
    proximity,
    proximity_allocations,
    sha256,
    version,
):
    _tables = {
        "comparison": comparison, "distribution": distribution,
        "frontier": frontier, "proximity": proximity,
        "origins": pd.concat([
            impact.assign(scenario="CAP25"),
            compare_origins(observed, protected, municipal).assign(scenario="CAP25 protected"),
        ], ignore_index=True),
    }
    output_paths = {}
    for _name, _table in _tables.items():
        _path = ROOT / f"metrics/equity_{_name}_{PERIOD}.csv"
        _table.to_csv(_path, index=False, encoding="utf-8-sig", float_format="%.8f")
        output_paths[_name] = _path
    _allocations = {"CAP25": cap25, "CAP25 protected": protected, **frontier_allocations, **proximity_allocations}
    _path = ROOT / f"results/equity_allocations_{PERIOD}.csv"
    pd.concat([s.assign(scenario=n) for n, s in _allocations.items()], ignore_index=True).to_csv(
        _path, index=False, encoding="utf-8-sig", float_format="%.8f",
    )
    output_paths["allocations"] = _path
    _path = ROOT / f"docs/equity_experiments_{PERIOD}.pdf"
    equity_figure.savefig(_path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    output_paths["figure"] = _path
    _svg = StringIO()
    equity_figure.savefig(_svg, format="svg", bbox_inches="tight", metadata={"Date": None})
    _sections = {
        "1 · Distribution of gains and losses": distribution,
        "1 · Largest increases in municipal mean": impact.nlargest(15, "change_km"),
        "2 · Municipal protection and controls": comparison,
        "3 · Efficiency–equity frontier": frontier,
        "4 · Municipal proximity guarantees": proximity,
    }
    _report = ROOT / "reports/equity_analysis.html"
    _report.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>CAP25 territorial equity</title>'
        '<style>body{font:16px system-ui;margin:2rem;color:#172b3a}section{overflow:auto;margin:2rem 0}'
        'table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap}'
        'svg{width:100%;height:auto}p{max-width:90ch}</style><h1>CAP25 territorial equity</h1>'
        '<p>Brazil · January–June 2026. Fixed observed admissions and provider municipalities. '
        'Integer flows; hypothetical capacities floor(1.25 × observed production). '
        'Efficiency uses rounded km; municipal means and proximity use original decimal km.</p>'
        '<p>Protection preserves each observed municipal mean. The frontier minimizes the worst municipal mean, '
        'then rounded total travel, within each budget; it does not impose protection. '
        'Proximity requires ceil(p × demand) nearby admissions in every origin. '
        'Thresholds are exploratory. Admissions in worsening origins are not identified individual losers. '
        'The analysis does not measure unmet need or universal coverage.</p>'
        + _svg.getvalue() + "".join(
            f"<section><h2>{name}</h2>{table.to_html(index=False, float_format=lambda x: f'{x:.4f}')}</section>"
            for name, table in _sections.items()
        ) + '</html>', encoding="utf-8",
    )
    output_paths["report"] = _report
    _source_paths = [
        *input_paths, ROOT / "notebooks/08_equity_analysis.py",
        *sorted((ROOT / "optimization").rglob("*.py")), *sorted((ROOT / "src").glob("*.py")),
    ]
    _metadata = {
        "period": PERIOD, "slack": SLACK, "epsilons": EPSILONS,
        "thresholds_km": THRESHOLDS_KM, "proximity_shares": PROXIMITY_SHARES,
        "n_admissions": sum(data.demand.values()), "n_origins": len(data.origins),
        "n_providers": len(data.providers),
        "total_capacity": sum(TransportScenario(slack=SLACK).capacities(data.observed_supply).values()),
        "efficiency_cost": "numpy.rint integer km; frontier budgets rounded down",
        "equity_distance": "original decimal km; municipal proximity counts rounded up",
        "integral_flows": True, "municipal_tolerance_km": 1e-5,
        "frontier_objective": "minimax municipal mean, then minimum rounded total distance",
        "frontier_includes_protection": False,
        "versions": {p: version(p) for p in ("marimo", "numpy", "pandas", "pyomo", "highspy")},
        "solver_options": {"threads": 1, "random_seed": 0, "mip_rel_gap": 0, "mip_abs_gap": 0},
        "unit": "observed admissions, not unique patients or unmet need",
        "inputs_sha256": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in _source_paths},
        "outputs_sha256": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in output_paths.values()},
    }
    _path = ROOT / f"metrics/equity_manifest_{PERIOD}.json"
    _path.write_text(json.dumps(_metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    output_paths["manifest"] = _path
    return (output_paths,)


@app.cell(hide_code=True)
def _(
    comparison,
    frontier,
    geographic_floor_km,
    impact,
    mo,
    output_paths,
    proximity,
):
    _protected = comparison.loc[comparison.scenario.eq("CAP25 protected")].iloc[0]
    mo.md(f"""
    ## FINDINGS AND EXPORTED ARTIFACTS

    - Published CAP25 worsens the municipal mean in **{impact.outcome.eq('worsened').sum()} origins**.
    - Municipal protection retains a **{_protected.distance_reduction_pct:.2f}%** reduction in total travel
      versus observed, at **{_protected.extra_model_distance_vs_cap25_pct:.2f}%** extra rounded travel versus CAP25.
    - The best frontier worst municipal mean is **{frontier.max_municipal_mean_km.min():.3f} km**;
      the geographic lower bound is **{geographic_floor_km:.3f} km**.
    - **{proximity.feasible.sum()} of {len(proximity)}** tested proximity policies are feasible.

    Feasibility and improvements refer to this fixed provider network and the
    hypothetical CAP25 capacities. They do not establish actual service readiness,
    absence of waiting lists, clinical equivalence or population coverage.

    """ + "\n".join(f"- `{p.name}`" for p in output_paths.values()))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
