"""Linear percentiles from aggregated admission frequencies"""

import numpy as np


def frequency_percentiles(values, counts, percentiles):
    """
    Match np.percentile on a repeated sample without expanding frequencies.

    Counts must be nonnegative integers with a positive total. For each percentile,
    cumulative counts locate the neighbours of (N - 1) * q in the conceptual sorted sample.
    """

    values      = np.asarray(values     )
    counts      = np.asarray(counts     )
    percentiles = np.asarray(percentiles, dtype=float)

    active = counts > 0

    values = values    [active]
    order  = np.argsort(values)
    values = values    [order ]

    cumulative = np.cumsum(
        counts[active][order],

        dtype=np.int64,
    )

    n = int(cumulative[-1])

    positions = (n - 1) * (percentiles / 100)
    lower     = np.floor  (positions        ).astype(np.int64)
    upper     = np.minimum(lower + 1, n - 1 )

    previous = values[np.searchsorted(cumulative, lower, side="right")]
    next_    = values[np.searchsorted(cumulative, upper, side="right")]

    fraction = positions - lower
    delta    = next_     - previous

    result = np.where(
        fraction >= 0.5,
        next_    - delta * (1 - fraction),
        previous + delta *      fraction ,
    )

    if np.isnan(values[-1]):
        result = np.full(percentiles.shape, np.nan)

    return result
