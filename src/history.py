"""Historical SIH acquisition using the shared selection and municipality rules"""

import re

import pandas as pd

from pathlib  import Path
from datetime import datetime, timezone

from .datasus   import list_files   , \
                       extract_files
from .municipal import ADMISSION_COLUMNS
from .sih       import prepare_catalog       , \
                       process_admissions    , \
                       load_municipalities   , \
                       consolidate_admissions


HOST      = "ftp.datasus.gov.br"
DIRECTORY = "/dissemin/publicos/SIHSUS/200801_/Dados"

STATES  = "AC AL AM AP BA CE DF ES GO MA MG MS MT PA PB PE PI PR RJ RN RO RR RS SC SE SP TO".split()
COLUMNS = [
    "N_AIH"    ,
    "CNES"     ,
    "IDENT"    ,
    "PROC_REA" ,
    "MUNIC_RES",
    "MUNIC_MOV",
    "DT_INTER" ,
    "DT_SAIDA" ,
    "ANO_CMPT" ,
    "MES_CMPT" ,
]


def clean_admissions(raw, years, procedures):
    """Select the latest normal AIH globally, then retain requested admission years"""

    raw = (
        raw
        .astype("string")
        .apply (lambda column: column.str.strip())
    )

    selected = raw["PROC_REA"].isin(procedures)

    data, normal_count, duplicate_count, outside_count = process_admissions(
        raw.loc[selected],

        f"{min(years)}-01-01",
        f"{max(years)}-12-31",
    )

    data["admission_year"] = data["admission_date"].dt.year
    requested              = data["admission_year"].isin(years)

    outside_count += int((~requested).sum())

    data = (
        data
        .loc[requested]
        .sort_values(
            [
                "DT_INTER",
                "N_AIH"   ,
                "CNES"    ,
            ]
        )
        .reset_index(drop=True)
    )

    if data.empty:
        raise ValueError("No admissions in the requested years.")

    return data, {
        "raw_selected_rows" : len(raw),

        "excluded_other_procedures"        : int((~selected).sum())               ,
        "excluded_nonregular_aih"          : int(  selected .sum()) - normal_count,
        "duplicate_presentations_removed"  : duplicate_count,
        "excluded_outside_admission_years" : outside_count  ,

        "admissions" : len(data),
    }


def fetch_admissions(
    root       : Path,
    years      : tuple[int, ...],
    procedures : tuple[str, ...],

    *,

    processing_cutoff="2026-09",
    workers          =4        ,
):
    """
    Return selected admissions, source catalog and explicit processing coverage

    All 27 UFs are required for each requested admission month. Available later
    processing files are also selected to capture resubmissions before the cutoff.
    Only filtered Parquet persists; new DBC/DBF files are temporary.
    """

    years      = tuple(sorted(set(years     )))
    procedures = tuple(sorted(set(procedures)))

    if not years or any(not isinstance(y, int) or isinstance(y, bool) or not 2008 <= y <= 2099 for y in years):
        raise ValueError("Admission years must be integers between 2008 and 2099.")

    if not procedures or any(not isinstance(p, str) or not re.fullmatch(r"\d{10}", p) for p in procedures):
        raise ValueError("Procedures must be ten-digit SIGTAP strings.")

    if not isinstance(processing_cutoff, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", processing_cutoff):
        raise ValueError("Processing cutoff must use YYYY-MM.")

    if processing_cutoff < f"{max(years)}-12":
        raise ValueError("Processing cutoff must cover all requested admission years.")

    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("Worker count must be a positive integer.")

    files = list_files(
        HOST     ,
        DIRECTORY,
        STATES   ,

        f"{min(years)}-01",
        processing_cutoff ,
    )

    required_months = [
        f"{year}-{month:02d}"
        for year  in years
        for month in range(1, 13)
    ]

    catalog, _ = prepare_catalog(
        files ,
        STATES,

        f"{min(years)}-01",
        f"{max(years)}-12",

        required_months=required_months,
    )
    available = set(
        zip(catalog["reference_month"], catalog["state"])
    )

    coverage = [
        {
            "reference_month" : str(month),
            "missing_states"  : [
                state
                for state in STATES
                if (str(month), state) not in available
            ],
        }
        for month in pd.period_range(f"{min(years)}-01", processing_cutoff, freq="M")
    ]

    raw = extract_files(
        catalog,

        Path(root) / ".cache/sih_prostatectomy",
        HOST      ,
        DIRECTORY ,
        procedures,
        COLUMNS   ,

        filtered_cache=Path(root) / ".cache/sih_history",
        keep_dbc      =False  ,
        workers       =workers,
    )

    admissions, audit = clean_admissions(raw, years, procedures)

    audit.update(
        extraction_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),

        years            =list(years     ) ,
        procedures       =list(procedures) ,
        processing_cutoff=processing_cutoff,

        source      =f"ftp://{HOST}{DIRECTORY}"     ,
        source_files=len(catalog                   ),
        source_bytes=int(catalog["file_size"].sum()),

        processing_coverage=coverage,
    )

    return admissions, catalog, audit


def prepare_admissions(admissions, root, municipalities=None):
    """
    Return canonical admissions, cohort coverage and every unmapped SIH pair

    The canonical columns match src.municipal.read_admissions, with an additional
    year field. Unmapped admissions remain visible in coverage and exclusions.
    """

    reference = (
        load_municipalities(root)
        if   municipalities is None
        else municipalities
    )

    consolidated = consolidate_admissions(admissions, municipalities=reference)

    consolidated["year"] = consolidated["admission_date"].dt.year

    located = consolidated[[
        "residence_ibge_code", "residence_municipality",
        "procedure_ibge_code", "procedure_municipality",
    ]].notna().all(axis=1)

    groups   = [
        "year"          ,
        "procedure_code",
    ]

    coverage = (
        consolidated
        .groupby (groups      )
        .size    (            )
        .rename  ("n_selected")
        .to_frame()
    )

    coverage["n_located" ] = (
        consolidated
        .loc[located]
        .groupby(groups)
        .size   (      )
        .reindex(coverage.index, fill_value=0)
    )

    coverage["n_excluded"] = coverage["n_selected"] - coverage["n_located"]

    coverage = coverage.reset_index().rename(
        columns={
            "procedure_code" : "procedure"
        }
    )

    excluded = consolidated.loc[
        ~located,
        [
            "year"          ,
            "procedure_code",
            "aih_number"    ,
            "cnes"          ,
        ]
    ].merge(
        admissions[
            [
                "N_AIH"    ,
                "CNES"     ,
                "MUNIC_RES",
                "MUNIC_MOV",
            ]
        ],

        left_on =["aih_number", "cnes"],
        right_on=["N_AIH"     , "CNES"],

        how     ="left"      ,
        validate="one_to_one",
    ).rename(columns={"procedure_code": "procedure"})

    excluded = excluded.groupby(
        [
            "year"     ,
            "procedure",
            "MUNIC_RES",
            "MUNIC_MOV",
        ],

        dropna=False,
    ).size().rename("n_admissions").reset_index()

    canonical = consolidated.loc         [located, [*ADMISSION_COLUMNS, "year"]          ] .rename     (columns=ADMISSION_COLUMNS)
    canonical = canonical   .sort_values(["admission_date", "admission_id", "hospital_id"]).reset_index(drop   =True)

    for field in canonical.columns.difference(["admission_date", "year"]):
        canonical[field] = canonical[field].astype("string")

    canonical.attrs.update(getattr(reference, "attrs", {}))

    return canonical, coverage, excluded
