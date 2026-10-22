"""Read the symmetric May 2024 Saldanha matrix without a persistent database"""

import duckdb

from pathlib import Path


DEFAULT_PARQUET_PATH = (
    Path(__file__).resolve().parents[1] / ".cache/saldanha/dist_brasil.parquet"
)

ROUTE_COLUMNS  = [
    "origin_code"     ,
    "destination_code",
    "distance_km"     ,
    "travel_time_min" ,
    "route_status"    ,
]

SOURCE_COLUMNS = {
    "orig",
    "dest",
    "dist",
    "dur" ,
}


def normalize_codes(codes):
    normalized = (
        codes
        .astype ("string")
        .str
        .strip  ()
        .str
        .replace(r"\.0+$", "", regex=True)
    )

    if not normalized.str.fullmatch(r"[0-9]{7}").fillna(False).all():
        raise ValueError("Municipality codes must contain seven IBGE digits.")

    return normalized


def _source(connection, path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Saldanha Parquet not found at {path}. Run: uv run python scripts/load_distances.py")

    relation = connection.read_parquet(str(path))

    missing  = SOURCE_COLUMNS - set(relation.columns)
    if missing:
        raise ValueError(f"Saldanha Parquet is missing columns: {sorted(missing)}")

    relation.create_view("saldanha")


def validate_parquet(path):
    """Check the readable schema and measures; source null estimates are allowed"""

    with duckdb.connect() as connection:
        _source(connection, path)

        rows, invalid = connection.execute(r"""
            SELECT count(*), count(*) FILTER (WHERE
                orig IS NULL OR dest IS NULL
                OR NOT regexp_full_match(
                    regexp_replace(trim(cast(orig AS VARCHAR)), '\.0+$', ''), '[0-9]{7}')
                OR NOT regexp_full_match(
                    regexp_replace(trim(cast(dest AS VARCHAR)), '\.0+$', ''), '[0-9]{7}')
                OR
                (
                    dist IS NOT NULL AND (try_cast(dist AS DOUBLE) IS NULL
                    OR NOT isfinite(try_cast(dist AS DOUBLE)) OR try_cast(dist AS DOUBLE) < 0)
                )
                OR
                (
                    dur IS NOT NULL AND (try_cast(dur AS DOUBLE) IS NULL
                    OR NOT isfinite(try_cast(dur AS DOUBLE)) OR try_cast(dur AS DOUBLE) < 0)
                )
            ) FROM saldanha
        """).fetchone()

    if not rows or invalid:
        raise ValueError("Saldanha Parquet is empty or contains invalid codes or distance/time measures.")


def query_distances(pairs, parquet_path=DEFAULT_PARQUET_PATH):
    """
    Return only requested pairs, in order, with explicit missing-data statuses.

    Source ``dist`` is in metres and ``dur`` is already in minutes. Source rows
    represent unordered pairs. Only original diagonal requests receive zeros;
    absent municipalities, absent pairs and null/zero estimates remain distinct.
    """

    requested = pairs[["origin_code", "destination_code"]].copy().reset_index(drop=True)

    for column in requested:
        requested[column] = normalize_codes(requested[column])

    requested["request_order"] = range(len(requested))

    with duckdb.connect() as connection:
        _source(connection, parquet_path)

        connection.register("requested", requested)

        result = connection.execute(r"""
            WITH source AS NOT MATERIALIZED (
                SELECT regexp_replace(trim(cast(orig AS VARCHAR)), '\.0+$', '')  AS orig,
                       regexp_replace(trim(cast(dest AS VARCHAR)), '\.0+$', '')  AS dest,
                       try_cast(dist AS DOUBLE) AS dist, try_cast(dur AS DOUBLE) AS dur
                FROM saldanha
            ), municipalities AS (
                SELECT orig AS code FROM source UNION SELECT dest AS code FROM source
            ), matches AS (
                SELECT least(orig, dest) AS origin, greatest(orig, dest) AS destination,
                    dist, dur, true AS found
                FROM source
                INNER JOIN (
                    SELECT DISTINCT least(origin_code, destination_code) AS origin,
                        greatest(origin_code, destination_code) AS destination
                    FROM requested
                ) AS needed ON least(orig, dest) = needed.origin
                    AND greatest(orig, dest) = needed.destination
            )
            SELECT r.origin_code, r.destination_code,
                CASE WHEN r.origin_code = r.destination_code THEN 0.0
                    WHEN m.dist > 0 AND isfinite(m.dist) THEN m.dist / 1000.0 END AS distance_km,
                CASE WHEN r.origin_code = r.destination_code THEN 0.0
                    WHEN m.dur > 0 AND isfinite(m.dur) THEN m.dur END AS travel_time_min,
                CASE WHEN r.origin_code = r.destination_code THEN 'intramunicipal'
                    WHEN o.code IS NULL OR d.code IS NULL THEN 'municipality_missing'
                    WHEN m.found IS NULL THEN 'pair_missing'
                    WHEN m.dist > 0 AND isfinite(m.dist) AND m.dur > 0 AND isfinite(m.dur)
                        THEN 'ok'
                    ELSE 'estimate_missing' END AS route_status
            FROM requested AS r
            LEFT JOIN matches AS m ON least(r.origin_code, r.destination_code) = m.origin
                AND greatest(r.origin_code, r.destination_code) = m.destination
            LEFT JOIN municipalities AS o ON r.origin_code = o.code
            LEFT JOIN municipalities AS d ON r.destination_code = d.code
            ORDER BY r.request_order
        """).df()

    if len(result) != len(requested):
        raise ValueError("Saldanha contains repeated unordered source pairs.")

    result[["origin_code", "destination_code"]] = result[
        [
            "origin_code"     ,
            "destination_code",
        ]
    ].astype("string")

    return result[ROUTE_COLUMNS]
