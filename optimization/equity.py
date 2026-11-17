"""Territorial equity experiments with fixed observed demand and CAP capacities"""

from decimal import Decimal, ROUND_CEILING

import numpy         as np
import pandas        as pd
import pyomo.environ as pyo

from .metrics          import flow_metrics
from .validation       import validate_solution
from .scenarios        import TransportScenario
from .models.transport import build_transport_model
from .solve            import extract_solution, solve_model, solver_options


TOLERANCE = 1e-5


def origin_means(solution):
    totals = solution.groupby("origin_code")[
        [
            "total_distance_km",
            "allocated_flow"   ,
        ]
    ].sum()

    return totals.total_distance_km / totals.allocated_flow


def compare_origins(observed, solution, municipal):
    """Paired municipal means; admission weights describe origins, not individuals"""

    table = pd.DataFrame({
        "demand" : observed.groupby("origin_code").allocated_flow.sum(),

        "observed_mean_km" : origin_means(observed),
        "scenario_mean_km" : origin_means(solution),
    })

    table["change_km"    ] = table.scenario_mean_km - table.observed_mean_km
    table["reduction_pct"] = -100 * table.change_km / table.observed_mean_km.replace(0, np.nan)

    table["outcome"] = np.select(
        [table.change_km.lt(-TOLERANCE), table.change_km.gt(TOLERANCE)],
        ["improved"                    , "worsened"                   ],

        default="unchanged",
    )

    reference = municipal.set_index("municipality_code")
    table     = table.join         (reference[["municipality", "state"]])

    table["region"] = table.state.map({
        **dict.fromkeys("AC AP AM PA RO RR TO"      .split(), "North"       ),
        **dict.fromkeys("AL BA CE MA PB PE PI RN SE".split(), "Northeast"   ),
        **dict.fromkeys("DF GO MT MS"               .split(), "Central-West"),
        **dict.fromkeys("ES MG RJ SP"               .split(), "Southeast"   ),
        **dict.fromkeys("PR RS SC"                  .split(), "South"       ),
    })

    table["observed_distance_band"] = pd.cut(
        table.observed_mean_km,
        [
            -np.inf,
            0      ,
            100    ,
            200    ,
            np.inf ,
        ],

        labels=[
            "local"        ,
            "(0, 100] km"  ,
            "(100, 200] km",
            ">200 km"      ,
        ],
    )

    return table.rename_axis("origin_code").reset_index()


def summarize_origins(impact, by=None):
    groups = impact.groupby(
        by,

        observed=True,
    ) if by else [("Brazil", impact)]

    rows = []

    for label, group in groups:
        row = {
            "scope"        : by or "national",
            "group"        : label           ,
            "n_origins"    : len(group             ),
            "n_admissions" : int(group.demand.sum()),
        }

        for outcome in ("improved", "unchanged", "worsened"):
            selected = group.loc[group.outcome.eq(outcome)]

            row[f"{outcome}_origins"   ] =       len(selected)
            row[f"{outcome}_origin_pct"] = 100 * len(selected) / len(group)

            row[f"{outcome}_origin_admissions"   ] = int  (selected.demand.sum())
            row[f"{outcome}_origin_admission_pct"] = 100 * selected.demand.sum() / group.demand.sum()

        row["mean_change_per_origin_km"   ] = group.change_km.mean()
        row["admission_weighted_change_km"] = np.average(group.change_km, weights=group.demand)

        rows.append(row)

    result = pd.DataFrame(rows)

    result[
        result.select_dtypes(include="float").columns
    ] = (
        result.select_dtypes(include="float").round(2)
    )

    return result

def distance_budget(reference_cost, epsilon):
    if epsilon < 0:
        raise ValueError("Efficiency tolerance must be nonnegative.")

    return int((Decimal(1) + Decimal(str(epsilon))) * int(reference_cost))


def solve_equity(
    data     ,
    distances,

    *,

    slack     =0.25 ,
    protection=None ,
    budget    =None ,
    minimax   =False,
    threshold =None ,
    share     =None ,
):
    """
    Solve integer extensions; minimax uses decimal km and a distance tie-break.

    Protection contains observed total decimal km by origin. Efficiency budgets
    use the original model's rounded integer km. Infeasibility is a result.
    """

    if (threshold is None) != (share is None):
        raise ValueError("Proximity requires both threshold and share.")

    if threshold is not None and (threshold < 0 or not 0 <= share <= 1):
        raise ValueError("Invalid proximity threshold/share.")

    if budget is not None and (budget < 0 or int(budget) != budget):
        raise ValueError("The rounded-distance budget must be a nonnegative integer.")

    scenario = TransportScenario(slack=slack)
    costs    = distances.to_dict()

    nearby, required = {}, {}

    row = {
        "feasible"        : False,
        "termination"     : ""   ,
        "blocked_origins" : ""   ,
    }

    if threshold is not None:
        for i in data.origins:
            nearby  [i] = [
                j
                for j in data.destinations_by_origin[i]
                if  costs[i, j] <= threshold
            ]

            required[i] = int(
                (Decimal(str(share)) * data.demand[i]).to_integral_value(rounding=ROUND_CEILING)
            )

        capacity = scenario.capacities(data.observed_supply)
        blocked  = [
            i
            for i in data.origins
            if sum(capacity[j] for j in nearby[i]) < required[i]
        ]

        if blocked:
            row.update(
                termination    ="infeasible_origin_proximity",
                blocked_origins=";".join(blocked)            ,
            )

            return None, row

    model = build_transport_model(data, scenario)

    upper_mean = None
    if minimax and budget is not None:
        solver = pyo.SolverFactory("appsi_highs")

        solver.options.update(solver_options(model))

        result = solver.solve(model)
        if result.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise RuntimeError("Optimal transportation LP required for arc bounds.")

        optimum_cost  = pyo.value(model.total_distance)
        reduced_costs = solver.get_reduced_costs()

        if len(reduced_costs) != len(model.x) or abs(optimum_cost - round(optimum_cost)) > TOLERANCE:
            raise RuntimeError("Incomplete LP reduced costs or nonintegral optimum.")

        allowance = budget - round(optimum_cost)
        if allowance < 0:
            row["termination"] = "infeasible"

            return None, row

        for variable, reduced_cost in reduced_costs.items():
            if abs(reduced_cost - round(reduced_cost)) > TOLERANCE:
                raise RuntimeError("Expected integer reduced costs from transportation simplex.")

            if round(reduced_cost) > 0:
                variable.setub(allowance // round(reduced_cost))

        upper_mean = max(
            sum(costs[i, j] * pyo.value(model.x[i, j]) for j in model.destinations[i]) / data.demand[i]
            for i in data.origins
        )

    for variable in model.x.values():
        variable.domain = pyo.NonNegativeIntegers

    travel = {
        i : pyo.quicksum(
            costs[i, j] * model.x[i, j]
            for j in model.destinations[i]
        )
        for i in data.origins
    }

    if protection is not None:
        model.protection = pyo.Constraint(model.I, rule=lambda m, i: travel[i] <= float(protection[i]))
    if budget is not None:
        model.budget     = pyo.Constraint(expr=model.total_distance.expr <= budget)
    if threshold is not None:
        model.proximity  = pyo.Constraint(
            model.I,

            rule=(
                lambda m, i: pyo.quicksum(m.x[i, j] for j in nearby[i]) >= required[i]
                if nearby[i]
                else pyo.Constraint.Feasible
            ),
        )

    if minimax:
        lower_bound = max(
            min(
                costs[i, j]
                for j in model.destinations[i]
            )
            for i in data.origins
        )

        model.worst_mean     = pyo.Var       (bounds=(lower_bound, upper_mean))
        model.municipal_mean = pyo.Constraint(
            model.I, rule=lambda m, i: travel[i] <= data.demand[i] * m.worst_mean,
        )

        model.total_distance.deactivate()

        model.equity_objective = pyo.Objective(expr=model.worst_mean)

    result = solve_model(model, allow_infeasible=True)

    row.update(
        termination  =str(result.solver.termination_condition),
        solver_status=str(result.solver.status               ),
    )

    if row["termination"] == "infeasible":
        return None, row

    if minimax:
        optimum = float(pyo.value(model.worst_mean))

        model.worst_mean      .setub     (optimum + TOLERANCE)
        model.equity_objective.deactivate()
        model.total_distance  .activate  ()

        solve_model(model)

        row["optimal_worst_mean_km"] = optimum

    objective = float(pyo.value(model.total_distance.expr))

    solution = extract_solution(
        model,
        data ,
        distances=distances,
    )

    row.update(
        validate_solution(
            solution ,
            data     ,
            scenario ,
            objective,
        )
    )
    row.update(
        feasible             =True,
        model_objective_km   =int  (round       (objective)      ),
        max_municipal_mean_km=float(origin_means(solution ).max()),
    )

    if budget is not None and objective > budget + TOLERANCE:
        raise RuntimeError("Efficiency budget exceeded.")

    if protection is not None:
        actual = solution.groupby("origin_code").total_distance_km.sum().reindex   (data.origins)
        excess = (actual - pd.Series(protection).reindex(data.origins)) / pd.Series(data.demand )

        if excess.max() > TOLERANCE:
            raise RuntimeError("Municipal protection failed.")

    if minimax and row["max_municipal_mean_km"] > optimum + 2 * TOLERANCE:
        raise RuntimeError("Equity tie-break lost the minimax optimum.")

    if threshold is not None:
        near = solution.loc[
            solution.distance_km.le(threshold)
        ].groupby("origin_code").allocated_flow.sum()

        counts = near.reindex(data.origins, fill_value=0)

        if (counts < pd.Series(required)).any():
            raise RuntimeError("Municipal proximity guarantee failed.")

        row["min_municipal_proximity_share"] = float((counts / pd.Series(data.demand)).min())

    return solution, row


def scenario_metrics(name, solution, observed, municipal, audit=None):
    impact = compare_origins(
        observed ,
        solution ,
        municipal,
    )

    summary = summarize_origins(impact).iloc[0].to_dict()

    result = {
        "scenario" : name,
        "feasible" : True,

        **flow_metrics(solution, float(observed.total_distance_km.sum())),

        **{
            key : value
            for key, value in summary.items()
            if  key not in {"scope", "group", "n_admissions"}
        },

        "max_municipal_mean_km" : float(impact.scenario_mean_km.max()),

        **(audit or {}),
    }

    return {
        k : (
            round(v, 2)
            if   isinstance(v, (float, np.floating))
            else v
        )

        for k, v in result.items()
    }
