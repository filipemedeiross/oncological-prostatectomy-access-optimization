"""Admission-weighted metrics evaluated with original distances"""

import numpy as np


def flow_metrics(solution, observed_total):
    n     = int  (solution.allocated_flow   .sum())
    total = float(solution.total_distance_km.sum())

    distances = np.repeat    (
        solution.distance_km   .to_numpy(             ),
        solution.allocated_flow.to_numpy(dtype="int64"),
    )
    quantiles = np.percentile(
        distances, [50, 75, 90, 95], method="linear"
    )

    metrics = {
        "n_admissions"      : n            ,
        "n_active_od_pairs" : len(solution),
        "total_distance_km" : total    ,
        "mean_distance_km"  : total / n,

        "max_distance_km"        : float (solution.distance_km.max()             ),
        "distance_reduction_pct" : 100 * (observed_total - total) / observed_total,

        **dict(
            zip(
                [
                    "median_distance_km",
                    "p75_distance_km"   ,
                    "p90_distance_km"   ,
                    "p95_distance_km"   ,
                ],
                map(float, quantiles),
            )
        ),
    }

    for level in ("municipality", "state"):
        same = int(
            solution.loc[
                solution[f"same_{level}"], "allocated_flow"
            ].sum()
        )

        metrics.update({
            f"n_same_{level}"      : same    ,
            f"n_outside_{level}"   : n - same,
            f"pct_same_{level}"    : 100 *  same      / n,
            f"pct_outside_{level}" : 100 * (n - same) / n,
        })

    return metrics
