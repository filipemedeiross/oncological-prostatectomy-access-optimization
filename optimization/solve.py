"""Solver execution and conversion of transportation variables to a tidy table."""

import numpy         as np
import pandas        as pd
import pyomo.environ as pyo

from .data import OptimizationData


def solver_options(model):
    integer = any(
        v.is_integer()
        for v in model.component_data_objects(
            pyo.Var    ,
            active=True,
        )
    )

    options = {
        "solver"      : (
            "choose"
            if   integer
            else "simplex"
        ),

        "threads"     : 1,
        "random_seed" : 0,
    }

    if integer:
        options.update({
            "mip_rel_gap"               : 0.0 ,
            "mip_abs_gap"               : 0.0 ,
            "mip_feasibility_tolerance" : 1e-6,
        })

    return options


def solve_model(model, *, allow_infeasible=False):
    solver = pyo.SolverFactory("appsi_highs")

    if not solver.available(exception_flag=False):
        raise RuntimeError("HiGHS is unavailable; install pyomo and highspy in this kernel.")

    solver.options.update(
        solver_options(model)
    )

    results     = solver .solve (
        model, load_solutions=False
    )
    termination = results.solver.termination_condition

    if (
        allow_infeasible and termination == pyo.TerminationCondition.infeasible
    ):
        return results

    if termination != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Optimal solution required; got {results.solver.termination_condition}.")

    model.solutions.load_from(results)

    return results


def extract_solution(
    model,
    data      : OptimizationData,
    tolerance : float = 1e-6    ,
) -> pd.DataFrame:
    if not 0 < tolerance < 0.5:
        raise ValueError("Flow tolerance must lie between zero and 0.5.")

    arcs = list    (model.A)
    flow = np.array(
        [
            pyo.value(model.x[arc])
            for arc in arcs
        ],
        dtype=float
    )

    if not np.isfinite(flow).all() or (flow < -tolerance).any():
        raise ValueError("Nonfinite or negative solver flow.")

    if not np.allclose(flow, np.rint(flow), rtol=0, atol=tolerance):
        raise ValueError("Solver flows are not integral within tolerance.")

    active   = flow > tolerance

    solution = pd.DataFrame(
        [
            arc
            for arc, keep in zip(arcs, active)
            if  keep
        ],
        columns=[
            "origin_code"     ,
            "destination_code",
        ]
    ).astype("string")

    solution["allocated_flow"   ] = np.rint(flow[active]).astype("int64")
    solution["distance_km"      ] = [
        data.distance[arc]
        for arc, keep in zip(arcs, active)
        if  keep
    ]
    solution["total_distance_km"] = solution.allocated_flow * solution.distance_km

    solution["same_municipality"] = solution.origin_code.eq (solution.destination_code)
    solution["same_state"       ] = solution.origin_code.map(data.origin_state        ).eq(
        solution.destination_code.map(
            data.provider_state
        )
    )

    return solution.sort_values(
        [
            "origin_code"     ,
            "destination_code",
        ]
    ).reset_index(drop=True)
