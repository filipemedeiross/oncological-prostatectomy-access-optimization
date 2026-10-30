"""Run historical cohorts with shared SIH, municipal and Saldanha conventions"""

import re
import json

import pandas as pd

from pathlib            import Path
from html               import escape
from importlib.metadata import version
from dataclasses        import dataclass
from tempfile           import NamedTemporaryFile

from datetime import datetime, timezone
from hashlib  import sha256  , file_digest

from optimization.data    import build_instance
from optimization.history import comparison_table, solve_scenarios, summarize

from src.routes    import prepare_routes     , route_policy
from src.history   import fetch_admissions   , prepare_admissions
from src.carbon    import carbon_estimates   , validate_carbon_parameters
from src.municipal import aggregate_municipal, attach_centroids, complete_od, load_boundaries, validate_od, write_csv

from optimization.history   import SCENARIOS
from src         .distances import DEFAULT_PARQUET_PATH
from src         .municipal import ADMISSION_COLUMNS, MUNICIPAL_COLUMNS, OD_COLUMNS


SCHEMA_VERSION = 1

DEFAULT_CARBON = {
    "factor_kg_co2e_per_pkm" : 0.02776,
    "legs"                   : 2      ,
    "travelers"              : 1      ,
    "distance_multiplier"    : 1.0    ,
    "price_per_tonne"        : None   ,
}

KEYS = [
    "origin_code"     ,
    "destination_code",
]


@dataclass
class HistoryResult:
    directory: Path
    annual   : pd.DataFrame
    summary  : pd.DataFrame
    coverage : pd.DataFrame
    manifest : dict


def _digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _signature(value):
    raw = json.dumps(
        value,

        sort_keys   =True ,
        ensure_ascii=False,
        allow_nan   =False,
    ).encode()

    return sha256(raw).hexdigest()


def _records(table):
    return json.loads(table.to_json(orient="records", date_format="iso"))


def _write(table, path):
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(
        dir   =path.parent,
        suffix=".parquet" ,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    try:
        table.to_parquet(temporary_path, index=False)

        temporary_path.replace(path)
    finally:
        temporary_path.unlink (missing_ok=True)


def _write_json(value, path):
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    raw = json.dumps(
        value,

        ensure_ascii=False,
        indent      =2    ,
        allow_nan   =False,
    )

    with NamedTemporaryFile(
        dir=path.parent,

        mode    ="w"    ,
        encoding="utf-8",
        delete  =False  ,
    ) as temporary:
        temporary.write(raw + "\n")

        temporary_path = Path(temporary.name)

    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink (missing_ok=True)


def _code_hashes(root):
    return {
        path.relative_to(root).as_posix() : _digest(path)
        for folder in       ("src", "optimization")
        for path   in sorted((root / folder).rglob("*.py"))
    }


def _selection(years, procedures, processing_cutoff=None):
    years      = tuple(sorted(set(years     )))
    procedures = tuple(sorted(set(procedures)))

    if not years or any(type(y) is not int or not 2008 <= y <= 2099 for y in years):
        raise ValueError("Years must be integers from 2008 through 2099.")

    if not procedures or any(not isinstance(p, str) or not re.fullmatch(r"\d{10}", p) for p in procedures):
        raise ValueError("Procedures must be ten-digit SIGTAP strings.")

    if processing_cutoff is not None and (
        not isinstance  (processing_cutoff       , str              ) or
        not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", processing_cutoff) or

        processing_cutoff < f"{max(years)}-12"
    ):
        raise ValueError("Processing cutoff must cover every requested full admission year.")

    return years, procedures


def load_history(directory, *, carbon=None):
    """Read a completed, hash-checked run; carbon assumptions do not rerun models"""

    directory = Path(directory)

    manifest = json.loads(
        (directory / "manifest.json").read_text()
    )

    for name, digest in manifest["output_sha256"].items():
        if _digest(directory / name) != digest:
            raise ValueError(f"Changed historical artifact: {directory / name}")

    annual  = pd.read_parquet(directory / "annual_statistics.parquet" )
    summary = pd.read_parquet(directory / "summary_statistics.parquet")

    audit   = json.loads(
        (directory / "audit.json").read_text()
    )

    assumptions = DEFAULT_CARBON | (carbon or {})
    annual      = carbon_estimates(annual , **assumptions)
    summary     = carbon_estimates(summary, **assumptions)

    manifest["carbon"] = {
        "assumptions" : assumptions,

        "currency"                 : "BRL"                                       ,
        "default_factor_reference" : "DESNZ 2025, Coach, factor 25_314_3146_11_1",
        "source"                   : "https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2025",
        "scope"                    : "Gross transport emissions only; potential equivalents, not certified credits."              ,
    }

    return HistoryResult(directory, annual, summary, pd.DataFrame(audit["coverage"]), manifest)


def analyze_history(
    root      ,
    admissions,
    years     ,
    procedures,

    *,

    parquet_path   =DEFAULT_PARQUET_PATH,
    carbon         =None,
    source_metadata=None,
    boundaries     =None,
):
    """
    Analyze canonical admissions, including an existing local study period.

    Each year/procedure keeps its own observed provider set and production.
    Missing alternatives stop the run, matching the current complete TXT model;
    routes and their audit remain available to diagnose the incomplete coverage.
    """

    root = Path(root).resolve()

    years, procedures = _selection(years, procedures)

    validate_carbon_parameters(**(DEFAULT_CARBON | (carbon or {})))

    columns = list(ADMISSION_COLUMNS.values())
    table   = admissions[columns].copy()

    if table.empty or table.isna().any().any() or table.duplicated(["admission_id", "hospital_id"]).any():
        raise ValueError("Historical admissions must be complete and deduplicated by AIH/CNES.")

    table["admission_date"] = pd.to_datetime(table.admission_date, errors="raise")
    table["year"          ] = table.admission_date.dt.year

    if not table.year.isin(years).all() or not table.procedure_code.isin(procedures).all():
        raise ValueError("Admissions outside the requested years/procedures.")

    for field, length in [
        ("admission_id"  , 13),
        ("hospital_id"   , 7 ),
        ("residence_code", 7 ),
        ("provider_code" , 7 ),
    ]:
        if not table[field].astype("string").str.fullmatch(rf"\d{{{length}}}").fillna(False).all():
            raise ValueError(f"Invalid historical admission identifier: {field}.")

        table[field] = table[field].astype("string")

    table = table.sort_values(
        [
            "admission_date",
            "admission_id"  ,
            "hospital_id"   ,
        ]
    ).reset_index(drop=True)

    expected = {
        (year, procedure)
        for year      in years
        for procedure in procedures
    }

    present = set(zip(table.year, table.procedure_code))

    if present != expected:
        raise ValueError(f"No located admissions for requested cohorts: {sorted(expected - present)}")

    boundaries = load_boundaries(root) if boundaries is None else boundaries

    municipal_parts, pairs = [], set()

    cohorts = table.groupby(["year", "procedure_code"], sort=True)

    for (year, procedure), cohort in cohorts:
        municipal = attach_centroids(aggregate_municipal(cohort), boundaries)

        municipal_parts.append(
            municipal.assign(
                year     =int(year),
                procedure=procedure,
            )
        )

        origins   = municipal.loc[municipal.demand         .gt(0), "municipality_code"]
        providers = municipal.loc[municipal.observed_supply.gt(0), "municipality_code"]

        pairs.update((i, j) for i in origins for j in providers)

    municipalities = pd.concat(municipal_parts, ignore_index=True)

    source_metadata = source_metadata or {}

    stable_source = json.loads(json.dumps(source_metadata))

    stable_source.get("selection"          , {}).pop("extraction_utc", None)
    stable_source.get("municipal_reference", {}).pop("reference_file", None)

    identity = {
        "schema_version" : SCHEMA_VERSION,

        "years"      : list(years     ),
        "procedures" : list(procedures),

        "admissions_sha256" : sha256(table         .to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
        "municipal_sha256"  : sha256(municipalities.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),

        "distance" : {
            "method"            : "saldanha",
            "network_reference" : "2024-05" ,
            "source"            : "https://zenodo.org/records/11400243",

            "parquet_sha256"   : _digest(parquet_path),
            "distance_unit"    : "km" ,
            "duration_unit"    : "min",
            "symmetric_source" : True ,

            "model_rounding" : "numpy.rint (half to even)"     ,
            "evaluation"     : "original unrounded distance_km",

            "route_policy" : route_policy(),
        },

        "source"      : stable_source     ,
        "code_sha256" : _code_hashes(root),

        "software"  : {
            name : version(name)
            for name in (
                "pandas"   ,
                "numpy"    ,
                "pyarrow"  ,
                "duckdb"   ,
                "pysus"    ,
                "pyomo"    ,
                "highspy"  ,
                "geopandas",
            )
        },

        "scenarios" : {
            key : {
                "label" : label,

                "slack"                : scenario.slack                if scenario else None,
                "max_interstate_share" : scenario.max_interstate_share if scenario else None,
            }
            for key, (label, scenario) in SCENARIOS.items()
        },
    }

    signature = _signature(identity)

    directory = root / ".cache/history" / f"{min(years)}_{max(years)}_{signature[:16]}"

    if (directory / "manifest.json").is_file():
        return load_history(directory, carbon=carbon)

    directory.mkdir(parents=True, exist_ok=True)

    _write(table         , directory / "admissions.parquet")
    _write(municipalities, directory / "municipal.parquet" )

    if (directory / "routes.parquet").is_file() and (directory / "audit.json").is_file():
        routes = pd.read_parquet(directory / "routes.parquet")
    else:
        requests = pd.DataFrame(
            sorted(pairs),

            columns=KEYS,
        ).astype("string")

        routes, proxy_audit = prepare_routes(requests, parquet_path)

        _write(routes, directory / "routes.parquet")

        coverage = source_metadata.get("coverage") or _records(
            table
            .groupby(
                [
                    "year"          ,
                    "procedure_code",
                ]
            )
            .size       ()
            .rename     ("n_selected")
            .reset_index()
            .rename     (columns  ={"procedure_code": "procedure"}             )
            .assign     (n_located=lambda frame: frame.n_selected, n_excluded=0)
        )

        _write_json(
            {
                "coverage" : coverage,

                "excluded_municipalities" : source_metadata.get("excluded_municipalities", []),
                "source_catalog"          : source_metadata.get("source_catalog"         , []),
                "selection"               : source_metadata.get("selection"              , {}),

                "route_status_counts" : {
                    str(k) : int(v)
                    for k, v in routes.route_status.value_counts().items()
                },

                "proxy_audit" : _records(proxy_audit),
            },

            directory / "audit.json",
        )

    metric_parts = []
    flow_parts   = []

    checkpoint_dir = directory / "checkpoints"

    for (year, procedure), cohort in cohorts:
        municipal = municipalities.loc[
            municipalities.year     .eq(year     ) &
            municipalities.procedure.eq(procedure),

            MUNICIPAL_COLUMNS
        ]

        od = complete_od(
            cohort   ,
            municipal,
        ).merge(
            routes,

            on      =KEYS        ,
            validate="one_to_one",
        )[OD_COLUMNS]

        validate_od(od, municipal, cohort)

        data        = build_instance(od, municipal)
        metric_path = checkpoint_dir / f"{year}_{procedure}_metrics.parquet"
        flow_path   = checkpoint_dir / f"{year}_{procedure}_flows.parquet"

        if metric_path.is_file() and flow_path.is_file():
            annual_metrics = pd.read_parquet(metric_path)
            annual_flows   = pd.read_parquet(flow_path  )
        else:
            annual_metrics, solutions = solve_scenarios(data, od)

            annual_metrics = annual_metrics.assign(
                year     =int(year),
                procedure=procedure,
            )

            annual_flows = pd.concat(
                [
                    flow.assign(
                        year     =int(year),
                        procedure=procedure,
                        scenario =scenario ,
                    )
                    for scenario, flow in solutions.items()
                ],

                ignore_index=True,
            )

            _write(annual_flows  , flow_path  )
            _write(annual_metrics, metric_path)

        metric_parts.append(annual_metrics)
        flow_parts  .append(annual_flows  )

        failed = annual_metrics.loc[annual_metrics.status.ne("optimal"), "scenario"].tolist()

        print(f"{year}/{procedure}: {len(cohort)} admissions; infeasible scenarios: {failed}", flush=True)

    metrics = pd.concat(metric_parts, ignore_index=True)
    flows   = pd.concat(flow_parts  , ignore_index=True)

    summary = summarize(
        metrics   ,
        flows     ,
        years     ,
        procedures,
    )

    _write(flows  , directory / "flows.parquet"             )
    _write(metrics, directory / "annual_statistics.parquet" )
    _write(summary, directory / "summary_statistics.parquet")

    outputs = (
        "admissions.parquet"        ,
        "municipal.parquet"         ,
        "routes.parquet"            ,
        "flows.parquet"             ,
        "annual_statistics.parquet" ,
        "summary_statistics.parquet",

        "audit.json",
    )

    manifest = identity | {
        "source"        : source_metadata,
        "run_signature" : signature      ,

        "generated_at_utc" : datetime.now(timezone.utc).isoformat(timespec="seconds"),

        "optimization_unit" : "year/procedure"                                                     ,
        "interpretation"    : "Historical flows evaluated on the fixed May 2024 reference network.",

        "output_sha256" : {
            name : _digest(directory / name)
            for name in outputs
        },
    }

    _write_json(manifest, directory / "manifest.json")

    for path in checkpoint_dir.glob("*.parquet"):
        path.unlink()

    if checkpoint_dir.is_dir():
        checkpoint_dir.rmdir()

    return load_history(directory, carbon=carbon)


def run_history(
    root,

    years     =tuple(range(2016, 2026)),
    procedures=("0416010121",)         ,

    *,

    processing_cutoff="2026-09"           ,
    workers          =4                   ,
    parquet_path     =DEFAULT_PARQUET_PATH,
    carbon           =None                ,
    refresh          =False               ,
):
    """
    Acquire/reuse a frozen SIH snapshot, then analyze annual cohorts.

    ``refresh=True`` checks the FTP again for revised source files. Otherwise a
    completed selection snapshot is reused, even when offline.
    """

    root = Path(root).resolve()

    years, procedures = _selection(
        years            ,
        procedures       ,
        processing_cutoff,
    )

    validate_carbon_parameters(**(DEFAULT_CARBON | (carbon or {})))

    if type(workers) is not int or workers < 1:
        raise ValueError("Workers must be a positive integer.")

    if not Path(parquet_path).is_file():
        raise FileNotFoundError("Saldanha matrix missing. Run: uv run python scripts/load_distances.py")

    source_code         = {name : _digest(root / name) for name in ("src/datasus.py", "src/sih.py", "src/history.py")}
    extraction_software = {name : version(name       ) for name in ("pandas", "pysus", "pyreaddbc", "pyarrow"       )}

    selection = {
        "schema_version" : SCHEMA_VERSION,

        "years"             : years              ,
        "procedures"        : procedures         ,
        "processing_cutoff" : processing_cutoff  ,
        "code_sha256"       : source_code        ,
        "software"          : extraction_software,
    }

    snapshot = root / ".cache/sih_history/snapshots" / _signature(selection)[:16]

    data_path  = snapshot / "admissions.parquet"
    audit_path = snapshot / "audit.json"

    if not refresh and data_path.is_file() and audit_path.is_file():
        metadata = json.loads(audit_path.read_text())

        if _digest(data_path) != metadata["admissions_sha256"]:
            raise ValueError("Changed historical source snapshot; use --refresh to reacquire it.")

        raw = pd.read_parquet(data_path)
    else:
        raw, catalog, audit = fetch_admissions(
            root      ,
            years     ,
            procedures,

            processing_cutoff=processing_cutoff,
            workers          =workers          ,
        )

        _write(raw, data_path)

        metadata = {
            "selection"           : audit,
            "source_catalog"      : _records(catalog  ),
            "extraction_software" : extraction_software,
            "admissions_sha256"   : _digest (data_path),
        }

        _write_json(metadata, audit_path)

    admissions, coverage, excluded = prepare_admissions(raw, root)

    metadata = metadata | {
        "coverage"                : _records(coverage        ),
        "excluded_municipalities" : _records(excluded        ),
        "municipal_reference"     : dict    (admissions.attrs),
    }

    return analyze_history(
        root      ,
        admissions,
        years     ,
        procedures,

        parquet_path   =parquet_path,
        carbon         =carbon      ,
        source_metadata=metadata    ,
    )


def export_history(result, root):
    """Explicitly publish two consolidated tables, a manifest and one HTML report"""

    root   = Path(root)
    folder = root / "metrics/"

    write_csv(result.annual , folder / "annual_statistics.csv" )
    write_csv(result.summary, folder / "summary_statistics.csv")

    report = root / "reports" / f"history.html"

    report.parent.mkdir(parents=True, exist_ok=True)

    content = (
        '<!doctype html><html lang="pt-BR"><meta charset="utf-8">'
        '<title>Historical surgical flows</title><body>'
        '<h1>Fluxos cirúrgicos históricos</h1>'
        '<p>Fluxos históricos avaliados na rede de referência de maio de 2024. '
        'Produção observada é a hipótese de capacidade. Métricas e carbono usam distâncias originais.</p>'
        + comparison_table(result.summary).to_html(index=False, border=0)
        + '<h2>Cobertura</h2>' + result.coverage.to_html(index=False, border=0)
        + '<h2>Emissões potenciais de transporte</h2>'
        + result.summary[["scenario", "status", "transport_tco2e", "avoided_tco2e", "potential_value"]]
          .rename(columns={"potential_value": "Valor potencial (BRL)"}).to_html(index=False, border=0)
        + '<h2>Hipóteses de carbono</h2><pre>'
        + escape(json.dumps(result.manifest["carbon"], ensure_ascii=False, indent=2)) + '</pre>'
        + '<p>Estimativa bruta de transporte; não representa créditos certificados.</p></body></html>'
    )

    report.write_text(content, encoding="utf-8")

    manifest = result.manifest | {
        "cache_directory" : str(result.directory),
        "export_sha256"   : {
            path.relative_to(root).as_posix() : _digest(path)
            for path in (
                folder / "annual_statistics.csv" ,
                folder / "summary_statistics.csv",
                report,
            )
        }
    }

    _write_json(manifest, folder / "manifest.json")

    return folder, report
