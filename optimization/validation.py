"""Check extracted allocations against input data, independently of Pyomo."""

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
    keys     = [
        "origin_code"     ,
        "destination_code",
    ]
    required = keys + [
        "allocated_flow"   ,
        "distance_km"      ,
        "total_distance_km",
        "same_municipality",
        "same_state"       ,
    ]

    _require(
        0 < tolerance < 0.5              and
        np.isfinite(objective_tolerance) and
        objective_tolerance > 0,

        "Invalid validation tolerances."
    )

    if termination_condition is not None:
        _require(str(termination_condition) == "optimal", "Solver termination is not optimal.")
    if solver_status         is not None:
        _require(str(solver_status        ) == "ok"     , "Solver status is not ok."          )

    _require(set(required) <= set(solution.columns), "Missing solution columns.")

    _require(not solution[required].isna().any().any(), "Incomplete solution.")
    _require(not solution.duplicated(keys)      .any(), "Duplicate OD pairs." )

    _require(solution.origin_code     .isin(data.origins  ).all(), "Origin outside I."     )
    _require(solution.destination_code.isin(data.providers).all(), "Destination outside J.")

    arcs          = list(solution[keys].itertuples(index=False, name=None))
    feasible_arcs = set (data.arcs)

    _require(
        all(
            arc in feasible_arcs
            for arc in arcs
        ),

        "Flow outside feasible arcs A."
    )

    flow = solution.allocated_flow.to_numpy(dtype=float)

    _require(   np.isfinite(flow).all(), "Nonfinite flow.")
    _require((flow >= -tolerance).all(), "Negative flow." )

    _require(np.allclose(flow, np.rint(flow), rtol=0, atol=tolerance), "Nonintegral flow.")

    _require((flow > tolerance).all(), "Only active OD pairs may be exported.")

    _require(
        np.isclose(
            flow.sum()               ,
            sum(data.demand.values()),
            rtol=0        ,
            atol=tolerance,
        ),

        "Total allocation differs from demand."
    )

    origin_totals = solution.groupby(
        "origin_code"
    ).allocated_flow.sum().reindex(data.origins, fill_value=0)

    demand = pd.Series(data.demand).reindex(data.origins)

    _require(
        np.allclose(origin_totals, demand, rtol=0, atol=tolerance), "Origin demand changed."
    )

    provider_totals = solution.groupby(
        "destination_code"
    ).allocated_flow.sum().reindex(data.providers, fill_value=0)

    capacity = pd.Series(
        {
            j : scenario.capacity(
                data.observed_supply[j]
            )
            for j in data.providers
        }
    )

    if check_capacity:
        _require(
            (provider_totals <= capacity + tolerance).all(), "Provider capacity exceeded."
        )

    if check_capacity and scenario.slack == 0:
        observed = pd.Series(data.observed_supply).reindex(data.providers)

        _require(
            np.allclose(
                provider_totals,
                observed       ,
                rtol=0         ,
                atol=tolerance ,
            ),

            "Zero-slack provider totals differ from observed supply."
        )

    distance = np.array([data.distance[arc] for arc in arcs])
    weighted = flow * distance

    _require(np.array_equal(solution.distance_km      , distance                                  ), "Solution distances differ from input distances.")
    _require(np.allclose   (solution.total_distance_km, weighted, rtol=0, atol=objective_tolerance), "Incorrect demand-weighted distances.")

    _require(solution.same_municipality.eq(solution.origin_code.eq (solution.destination_code)).all(), "Incorrect same-municipality flags.")
    _require(solution.same_state       .eq(solution.origin_code.map(data    .origin_state     ).eq (
        solution.destination_code.map(data.provider_state))).all(),

        "Incorrect same-state flags."
    )

    interstate = np.array(
        [
            data.origin_state[i] != data.provider_state[j]
            for i, j in arcs
        ]
    )
    interstate_admissions = float                    (flow[interstate].sum   () )
    interstate_limit      = scenario.interstate_limit(sum(data.demand .values()))

    if interstate_limit is not None:
        _require(
            interstate_admissions <= interstate_limit + tolerance,
            "Interstate admissions exceed the scenario limit."   ,
        )

    reconstructed = float(weighted.sum())

    _require(
        np.isfinite(objective                                                 ) and
        np.isclose (reconstructed, objective, rtol=0, atol=objective_tolerance),

        "Reconstructed objective differs from the solver objective."
    )

    return {
        "max_demand_residual"   : float(abs(origin_totals - demand                ).max()),
        "max_integrality_error" : float(abs(flow          - np.rint(flow)         ).max()),
        "max_capacity_excess"   : float((provider_totals  - capacity).clip(lower=0).max()) if check_capacity else None,

        "objective_error_km"    : abs(reconstructed - objective),
        "n_saturated_providers" : int(
            np.isclose(
                provider_totals,
                capacity       ,

                rtol=0        ,
                atol=tolerance,
            ).sum()
        ) if check_capacity else None,

        "interstate_admissions"     : interstate_admissions,
        "max_interstate_admissions" : interstate_limit     ,

        "interstate_constraint_active" : bool(
            np.isclose(
                interstate_admissions,
                interstate_limit     ,

                rtol=0        ,
                atol=tolerance,
            )
        ) if interstate_limit is not None else None,
    }
