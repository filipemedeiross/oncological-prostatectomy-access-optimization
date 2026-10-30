"""Scenario estimates of gross transport emissions using original distances"""

import numpy as np


def validate_carbon_parameters(
    *,

    factor_kg_co2e_per_pkm,

    legs               =2   ,
    travelers          =1   ,
    distance_multiplier=1.0 ,
    price_per_tonne    =None,
):
    """Reject invalid assumptions before acquisition or optimization begins"""

    parameters = [
        factor_kg_co2e_per_pkm,
        legs                  ,
        travelers             ,
        distance_multiplier   ,
    ]

    try:
        valid = all(
            np.isfinite(value) and value > 0
            for value in parameters
        )

        valid_price = price_per_tonne is None or (np.isfinite(price_per_tonne) and price_per_tonne >= 0)
    except (TypeError, ValueError):
        raise ValueError("Emission assumptions and carbon price must be finite numbers.") from None

    if not valid:
        raise ValueError("Emission factor, trip legs, travelers and distance adjustment must be positive and finite.")
    if not valid_price:
        raise ValueError("Carbon price must be nonnegative and finite.")


def carbon_estimates(
    metrics,

    *,

    factor_kg_co2e_per_pkm,

    legs               =2   ,
    travelers          =1   ,
    distance_multiplier=1.0 ,
    price_per_tonne    =None,
):
    """
    Convert original distances to gross transport tonnes, not issued credits.

    Surgical/infrastructure emissions are outside this calculation. Incomplete
    rows remain missing, and negative distance savings remain negative.
    """

    validate_carbon_parameters(
        factor_kg_co2e_per_pkm=factor_kg_co2e_per_pkm,
        legs                  =legs                  ,
        travelers             =travelers             ,
        distance_multiplier   =distance_multiplier   ,
        price_per_tonne       =price_per_tonne       ,
    )

    result = metrics.copy()

    multiplier = factor_kg_co2e_per_pkm * legs * travelers * distance_multiplier / 1000

    result["transport_tco2e"] = result.total_distance_km * multiplier
    result["avoided_tco2e"  ] = result.absolute_gain_km  * multiplier
    result["potential_value"] = result.avoided_tco2e     * (price_per_tonne if price_per_tonne is not None else np.nan)

    return result
