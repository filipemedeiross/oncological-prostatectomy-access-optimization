"""Transportation LP, or integer transportation with a global interstate cap."""

import pyomo.environ as pyo

from ..data      import OptimizationData
from ..scenarios import TransportScenario


def add_sets(model, data):
    model.I = pyo.Set(                                   initialize=data.origins  , ordered=True)
    model.J = pyo.Set(                                   initialize=data.providers, ordered=True)
    model.A = pyo.Set(dimen=2, within=model.I * model.J, initialize=data.arcs     , ordered=True)

    model.destinations = pyo.Set(model.I, initialize=data.destinations_by_origin)
    model.origins      = pyo.Set(model.J, initialize=data.origins_by_destination)


def add_parameters(model, data, scenario):
    model.D = pyo.Param(model.I, within=pyo.NonNegativeIntegers, initialize=data.demand)
    model.U = pyo.Param(model.J, within=pyo.NonNegativeIntegers, initialize={
        j : scenario.capacity(
            data.observed_supply[j]
        )
        for j in data.providers
    })

    model.d = pyo.Param(model.A, within=pyo.NonNegativeReals, initialize=data.distance)


def add_flow_variables(model, scenario):
    domain = (
        pyo.NonNegativeReals
        if   scenario.max_interstate_share is None
        else pyo     .NonNegativeIntegers
    )

    model.x = pyo.Var(model.A, domain=domain)


def add_demand_constraints(model):
    model.demand_balance = pyo.Constraint(
        model.I,

        rule=lambda m, i: pyo.quicksum(
            m.x[i, j]
            for j in m.destinations[i]
        ) == m.D[i]
    )


def add_capacity_constraints(model):
    model.provider_capacity = pyo.Constraint(
        model.J,

        rule=lambda m, j: pyo.quicksum(
            m.x[i, j]
            for i in m.origins[j]
        ) <= m.U[j]
    )


def add_interstate_constraint(model, data, scenario):
    limit = scenario.interstate_limit(
        sum(data.demand.values())
    )

    if limit is None:
        return

    model.A_inter = pyo.Set(
        dimen  =2,
        within =model.A,
        ordered=True   ,

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
                model.x[i, j]
                for i, j in model.A_inter
            ) <= model.max_interstate_admissions
            if   len(model.A_inter)
            else pyo.Constraint.Feasible
        )
    )


def add_distance_objective(model):
    model.total_distance = pyo.Objective(
        expr =pyo.quicksum(
            model.d[i, j] * model.x[i, j]
            for i, j in model.A
        ),
        sense=pyo.minimize,
    )


def build_transport_model(
    data     : OptimizationData ,
    scenario : TransportScenario,
):
    model = pyo.ConcreteModel(
        name="Capacity-constrained provider reallocation"
    )

    add_sets                (model, data          )
    add_parameters          (model, data, scenario)
    add_flow_variables      (model,       scenario)

    add_demand_constraints   (model)
    add_capacity_constraints (model)
    add_interstate_constraint(model, data, scenario)
    add_distance_objective   (model)

    return model
