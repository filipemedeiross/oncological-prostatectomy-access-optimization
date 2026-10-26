"""Check solver allocations against demand, capacity and integer model costs"""

import numpy  as np
import pandas as pd

from .data      import OptimizationData
from .scenarios import TransportScenario


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_solution(
    solution : pd.DataFrame     ,
    data     : OptimizationData ,
    scenario : TransportScenario,

    objective           : float,
    tolerance           : float = 1e-6,
    objective_tolerance : float = 1e-5,

    *,
    check_capacity        : bool = True,
    termination_condition        = None,
    solver_status                = None,
) -> dict:
    if termination_condition is not None:
        _require(str(termination_condition) == "optimal", "Solver termination is not optimal.")
    if solver_status         is not None:
        _require(str(solver_status        ) == "ok"     , "Solver status is not ok."          )

    flow              = solution.allocated_flow.to_numpy(dtype  =float)
    integrality_error = np.abs(flow - np.rint(flow)).max(initial=0    )

    _require(np.isfinite(flow).all() and (flow >= 0).all(), "Nonfinite or negative flow.")
    _require(integrality_error <= tolerance               , "Nonintegral flow."          )

    origin_totals   = solution.groupby("origin_code"     ).allocated_flow.sum().reindex(data.origins  , fill_value=0)
    provider_totals = solution.groupby("destination_code").allocated_flow.sum().reindex(data.providers, fill_value=0)

    demand   = pd.Series(data.demand)
    capacity = pd.Series(
        {
            j : scenario.capacity(data.observed_supply[j])
            for j in data.providers
        }
    )

    demand_residual = float((origin_totals   - demand  ).abs ()       .max())
    capacity_excess = float((provider_totals - capacity).clip(lower=0).max())

    _require(demand_residual <= tolerance, "Origin demand changed.")
    if check_capacity:
        _require(capacity_excess <= tolerance, "Provider capacity exceeded.")

    arcs = list(
        solution[
            [
                "origin_code"     ,
                "destination_code",
            ]
        ]
        .itertuples(
            index=False,
            name =None ,
        )
    )

    interstate = [
        data.origin_state[i] != data.provider_state[j]
        for i, j in arcs
    ]

    interstate_admissions = float(flow[interstate].sum())
    interstate_limit      = scenario.interstate_limit(sum(data.demand.values()))

    if interstate_limit is not None:
        _require(
            interstate_admissions <= interstate_limit + tolerance,
            "Interstate admissions exceed the scenario limit."   ,
        )

    reconstructed = float(flow @ np.array([data.distance[arc] for arc in arcs]))

    _require(
        np.isfinite(objective)
        and
        np.isclose (
            reconstructed,
            objective    ,

            rtol=0                  ,
            atol=objective_tolerance,
        ),

        "Reconstructed objective differs from the solver objective.",
    )

    return {
        "objective_error_km" : abs(reconstructed - objective),

        "n_saturated_providers" : (
            int(
                np.isclose(
                    provider_totals,
                    capacity       ,

                    rtol=0        ,
                    atol=tolerance,
                ).sum()
            )
            if check_capacity else None
        ),

        "interstate_admissions"        : interstate_admissions,
        "max_interstate_admissions"    : interstate_limit     ,
        "interstate_constraint_active" : (
            bool(
                np.isclose(
                    interstate_admissions,
                    interstate_limit     ,

                    rtol=0        ,
                    atol=tolerance,
                )
            )
            if interstate_limit is not None else None
        ),
    }
