"""Annual scenarios using the prepared OD routes and current transport model"""

import numpy         as np
import pandas        as pd
import pyomo.environ as pyo

from src.reallocation import observed_baseline, nearest_provider

from .data             import OptimizationData
from .scenarios        import TransportScenario

from .metrics          import flow_metrics
from .validation       import validate_solution
from .models.transport import build_transport_model
from .solve            import extract_solution, solve_model


KEYS = [
    "origin_code"     ,
    "destination_code",
]

SCENARIOS = {
    "observed" : ("Observado"     , None                        ),
    "s3_0102"  : ("S3 (0%; 1,02%)", TransportScenario(0, 0.0102)),
    "s3_0200"  : ("S3 (0%; 2,00%)", TransportScenario(0, 0.02  )),
    "s2_0"     : ("S2 (0%)"       , TransportScenario(0        )),
    "s2_5"     : ("S2 (5%)"       , TransportScenario(0.05     )),
    "s2_10"    : ("S2 (10%)"      , TransportScenario(0.1      )),
    "s2_20"    : ("S2 (20%)"      , TransportScenario(0.2      )),
    "s1"       : ("S1"            , None                        ),
}


def _metrics(solution, observed_total):
    """Keep the shared metrics, including a zero-distance observed cohort"""

    result = flow_metrics(
        solution, observed_total if observed_total else 1.0
    )

    result["absolute_gain_km"      ] = observed_total - result["total_distance_km"]
    result["distance_reduction_pct"] = (
        100 * result["absolute_gain_km"] / observed_total if observed_total else 0.0
    )

    return result


def _check_instance(data, od):
    """Reject mismatched annual margins, route universes and model costs"""

    indexed = od.set_index(KEYS, verify_integrity=True)

    if (
        len(indexed                          ) != len(data.origins  ) * len(data.providers) or
        set(indexed.index                    ) != set(data.arcs                           ) or
        set(indexed.index.get_level_values(0)) != set(data.origins                        ) or
        set(indexed.index.get_level_values(1)) != set(data.providers                      )
    ):
        raise ValueError("OD routes differ from the annual origin x provider instance.")

    distances = indexed  .distance_km
    values    = distances.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Every annual OD alternative requires a resolved distance.")

    expected_costs = np.rint (
        distances.loc[list(data.arcs)].to_numpy(dtype=float)
    )

    actual_costs   = np.array(
        [
            data.distance[arc]
            for arc in data.arcs
        ]
    )

    if not np.array_equal(expected_costs, actual_costs):
        raise ValueError("Model costs differ from the rounded original OD distances.")

    for key, counts in [
        ("origin_code"     , data.demand         ),
        ("destination_code", data.observed_supply),
    ]:
        expected = pd.Series (counts, dtype="int64"  ).sort_index()
        actual   = od.groupby(key).observed_flow.sum().sort_index()

        if set(actual.index) != set(expected.index) or not actual.reindex(expected.index).eq(expected).all():
            raise ValueError(f"Annual observed margins differ from the instance: {key}.")

    return distances


def _model_costs(solution, data):
    solution = solution.copy()

    solution["model_distance_km"] = [
        data.distance[arc]
        for arc in solution[KEYS].itertuples(index=False, name=None)
    ]

    return solution


def solve_scenarios(data: OptimizationData, od: pd.DataFrame):
    """
    Return metrics and positive flows for the eight annual scenarios.

    All scenarios use the same original OD routes. Capacity scenarios optimize
    integer-kilometre costs, while metrics retain the original route distances.
    Infeasible interstate scenarios remain explicit and are never relaxed.
    """

    baseline  = observed_baseline(od).rename(
        columns={
            "observed_flow" : "allocated_flow"
        }
    )
    distances = _check_instance  (data, od)

    baseline       = _model_costs(baseline, data)

    observed_total = float(
        baseline.total_distance_km.sum()
    )

    nearest        = _model_costs(
        nearest_provider(
            od, pd.Series(data.demand, dtype="int64")
        ),

        data,
    )

    state_demand = pd.Series(data.demand).groupby(pd.Series(data.origin_state)).sum()

    rows, solutions = [], {}

    for key, (_, scenario) in SCENARIOS.items():
        row = {
            "scenario" : key      ,
            "status"   : "optimal",

            "slack"                     : scenario.slack                                       if scenario else None,
            "max_interstate_share"      : scenario.max_interstate_share                        if scenario else None,
            "max_interstate_admissions" : scenario.interstate_limit(sum(data.demand.values())) if scenario else None,

            "minimum_interstate_required" : None,
        }

        if scenario is None:
            solution  = baseline.copy() if key == "observed" else nearest.copy()

            objective = float(
                solution.allocated_flow @ solution.model_distance_km
            )
        else:
            capacity = pd.Series({
                j : scenario.capacity(supply)
                for j, supply in data.observed_supply.items()
            })

            state_capacity = capacity.groupby(
                pd.Series(data.provider_state)
            ).sum()

            minimum = int(
                (
                    state_demand - state_capacity.reindex(state_demand.index, fill_value=0)
                )
                .clip(lower=0)
                .sum ()
            )

            row["minimum_interstate_required"] = minimum

            if row["max_interstate_admissions"] is not None and minimum > row["max_interstate_admissions"]:
                rows.append(
                    row | {
                        "status"               : "infeasible"            ,
                        "infeasibility_reason" : "state_capacity_deficit",
                    }
                )

                continue

            model  = build_transport_model(data , scenario)
            result = solve_model          (model, allow_infeasible=True)

            row["status"] = str(result.solver.termination_condition)

            if row["status"] == "infeasible":
                rows.append(row | {"infeasibility_reason": "solver"})

                continue

            objective = float(pyo.value(model.total_distance))

            solution  = extract_solution(
                model,
                data ,

                distances=distances,
            )

            solution = solution.merge(
                od[KEYS + ["travel_time_min", "route_status"]],

                on      =KEYS        ,
                how     ="left"      ,
                validate="one_to_one",
            )

        checks = validate_solution(
            solution                       ,
            data                           ,
            scenario or TransportScenario(),
            objective                      ,

            check_capacity=(key != "s1"),
        )

        rows.append(
            row                                  |
            { "model_objective_km" : objective } |
            _metrics(solution, observed_total)   |
            checks
        )

        solutions[key] = solution

    return pd.DataFrame(rows), solutions


def summarize(metrics, flows, years, procedures):
    """Recompute weighted period metrics only when every cohort is successful"""

    expected = {
        (int(year), procedure)
        for year      in years
        for procedure in procedures
    }

    if not expected:
        raise ValueError("At least one year and procedure are required.")

    groups = ["year", "procedure", "scenario"]

    metrics = metrics.loc[metrics.year.isin(years) & metrics.procedure.isin(procedures)].copy()
    flows   = flows  .loc[flows  .year.isin(years) & flows  .procedure.isin(procedures)].copy()

    if metrics.duplicated(groups).any() or flows.duplicated(groups + KEYS).any():
        raise ValueError("Repeated annual metrics or scenario OD flows.")

    if not metrics.scenario.isin(SCENARIOS).all() or not flows.scenario.isin(SCENARIOS).all():
        raise ValueError("Unknown historical scenario.")

    counts = flows.allocated_flow.to_numpy(dtype=float)
    values = flows.distance_km   .to_numpy(dtype=float)

    if (
        not np.isfinite(counts).all() or (counts <= 0).any() or
        not np.equal   (counts, np.rint(counts)      ).all() or
        not np.isfinite(values).all() or (values < 0 ).any() or
        not np.allclose(
            counts * values                              ,
            flows.total_distance_km.to_numpy(dtype=float),

            rtol=0   ,
            atol=1e-5,
        )
    ):
        raise ValueError("Historical flows contain invalid counts or weighted distances.")

    actual = flows.groupby(
        groups
    )[["allocated_flow", "total_distance_km"]].sum()

    successful = metrics.loc[metrics.status.eq("optimal")].set_index(groups)
    aligned    = actual .reindex(successful.index)

    if (
        not np.array_equal(aligned.allocated_flow   , successful.n_admissions                        ) or
        not np.allclose   (aligned.total_distance_km, successful.total_distance_km, rtol=0, atol=1e-5) or
        not actual.index.isin(successful.index).all()
    ):
        raise ValueError("Annual flows are absent or differ from their scenario metrics.")

    observed = successful.loc[
        successful.index.get_level_values("scenario") == "observed"
    ].reset_index().set_index(["year", "procedure"])

    for (year, procedure, _), row in successful.iterrows():
        if (year, procedure) in observed.index and row.n_admissions != observed.loc[(year, procedure), "n_admissions"]:
            raise ValueError("A scenario changed the observed number of admissions.")

    observed_periods = set(observed.index)

    baseline = float(
        flows.loc[flows.scenario.eq("observed"), "total_distance_km"].sum()
    )

    rows = []

    for key, (_, scenario) in SCENARIOS.items():
        part = metrics.loc[metrics.scenario.eq(key)]

        present = set(
            part
            .loc       [part.status.eq("optimal"), ["year", "procedure"]]
            .itertuples(index=False, name=None)
        )

        missing = sorted((expected - present) | (expected - observed_periods))

        row = dict.fromkeys(metrics.columns.difference(["year", "procedure"]), np.nan) | {
            "scenario" : key                                   ,
            "status"   : "incomplete" if missing else "optimal",

            "missing_periods"      : "; ".join(f"{year}/{procedure}" for year, procedure in missing),
            "slack"                : scenario.slack                if scenario else None,
            "max_interstate_share" : scenario.max_interstate_share if scenario else None,
        }

        if not missing:
            solution = flows.loc[flows.scenario.eq(key)].groupby(
                KEYS + [
                    "distance_km"      ,
                    "same_municipality",
                    "same_state"       ,
                ],

                as_index=False,
            )[["allocated_flow", "total_distance_km"]].sum()

            row.update(_metrics(solution, baseline))

            for field in [
                "model_objective_km"         ,
                "max_interstate_admissions"  ,
                "minimum_interstate_required",
            ]:
                if field in part:
                    row[field] = part[field].sum(min_count=1)

            if "objective_error_km" in part:
                row["objective_error_km"] = part.objective_error_km.max()

        rows.append(row)

    return pd.DataFrame(rows)


def comparison_table(summary):
    """Format the reference comparison table in Portuguese"""

    columns = {
        "scenario"               : "Cenário"                   ,
        "mean_distance_km"       : "Média (km)"                ,
        "max_distance_km"        : "Máxima (km)"               ,
        "distance_reduction_pct" : "Redução total"             ,
        "absolute_gain_km"       : "Ganho absoluto (km)"       ,
        "n_outside_state"        : "Internações interestaduais",
    }

    table = summary.reindex(
        columns=columns
    ).rename(
        columns=columns
    ).copy()

    table["Cenário"] = table["Cenário"].map(
        {
            key : value[0]
            for key, value in SCENARIOS.items()
        }
    )

    for column in table.columns[1:]:
        decimals = 0 if column == "Internações interestaduais" else 2

        table[column] = table[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:,.{decimals}f}".translate(str.maketrans(",.", ".,"))
        )

        if column == "Redução total":
            table[column] = table[column].map(lambda value: value + "%" if value != "—" else value)

    return table
