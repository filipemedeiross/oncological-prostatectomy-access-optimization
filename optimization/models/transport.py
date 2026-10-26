"""Transportation LP, or integer transportation with a global interstate cap."""

import pyomo.environ as pyo

from ..data      import OptimizationData
from ..scenarios import TransportScenario


def build_transport_model(
    data     : OptimizationData ,
    scenario : TransportScenario,
):
    model = pyo.ConcreteModel(
        name="Capacity-constrained provider reallocation"
    )

    model.I = pyo.Set(         initialize=data.origins  , ordered=True)
    model.J = pyo.Set(         initialize=data.providers, ordered=True)
    model.A = pyo.Set(dimen=2, initialize=data.arcs     , ordered=True)

    model.destinations = pyo.Set(model.I, initialize=data.destinations_by_origin)
    model.origins      = pyo.Set(model.J, initialize=data.origins_by_destination)

    model.D = pyo.Param(model.I, within=pyo.NonNegativeIntegers, initialize=data.demand)
    model.U = pyo.Param(
        model.J,

        within    =pyo.NonNegativeIntegers                                                ,
        initialize={j: scenario.capacity(data.observed_supply[j]) for j in data.providers},
    )
    model.d = pyo.Param(model.A, within=pyo.NonNegativeIntegers, initialize=data.distance)

    domain = (
        pyo.NonNegativeReals
        if   scenario.max_interstate_share is None
        else pyo     .NonNegativeIntegers
    )

    model.x = pyo.Var(model.A, domain=domain)

    model.demand_balance    = pyo.Constraint(
        model.I,
        rule=lambda m, i: pyo.quicksum(m.x[i, j] for j in m.destinations[i]) == m.D[i],
    )
    model.provider_capacity = pyo.Constraint(
        model.J,
        rule=lambda m, j: pyo.quicksum(m.x[i, j] for i in m.origins     [j]) <= m.U[j],
    )

    limit = scenario.interstate_limit(sum(data.demand.values()))
    if limit is not None:
        model.A_inter = pyo.Set(
            dimen=2,

            initialize=[
                (i, j)
                for i, j in data.arcs
                if  data.origin_state[i] != data.provider_state[j]
            ],
        )

        model.max_interstate_admissions = pyo.Param(
            within    =pyo.NonNegativeIntegers,
            initialize=limit                  ,
        )

        model.interstate_limit = pyo.Constraint(
            expr=(
                pyo.quicksum(
                    model.x[arc]
                    for arc in model.A_inter
                )
                <=
                model.max_interstate_admissions

                if   len(model.A_inter)
                else pyo.Constraint.Feasible
            )
        )

    model.total_distance = pyo.Objective(
        expr =pyo.quicksum(
            model.d[arc] * model.x[arc]
            for arc in model.A
        ),
        sense=pyo.minimize,
    )

    return model
