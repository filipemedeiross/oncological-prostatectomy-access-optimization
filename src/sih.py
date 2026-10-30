"""Coverage validation, SIH admission selection, and municipality identification"""

import json
import httpx

from pathlib  import Path
from hashlib  import sha256
from tempfile import NamedTemporaryFile

import pandas as pd


def prepare_catalog(
    files     ,
    states    ,
    start_date,
    end_date  ,

    *,

    required_months=None,
):
    """Organize the catalog and validate state/month coverage for the period"""

    catalog = pd.DataFrame(
        files,

        columns=[
            "filename"       ,
            "state"          ,
            "reference_month",
            "last_modified"  ,
            "file_size"      ,
        ],
    ).sort_values(
        [
            "reference_month",
            "state"          ,
        ]
    ).reset_index(drop=True)

    if catalog.duplicated(["reference_month", "state"]).any():
        raise ValueError("Repeated state/processing-month source files.")

    months = (
        pd.period_range(
            start_date,
            end_date  ,

            freq="M",
        ).astype(str)
        if   required_months is None
        else pd.Index(required_months, dtype="string")
    )

    coverage = pd.crosstab(
        catalog["reference_month"],
        catalog["state"          ],
    ).reindex(
        columns   =states,
        fill_value=0     ,
    )

    missing = coverage.reindex(months, fill_value=0).eq(0).stack()

    if missing.any():
        raise RuntimeError(
            f"Missing state/month combinations: {missing[missing].index.tolist()}"
        )

    return catalog, coverage


def process_admissions(raw_data, start_date, end_date):
    """Select standard AIHs, keep the latest submission, and filter admission dates"""

    raw_data = (
        raw_data
        .astype("string"                         )
        .apply (lambda column: column.str.strip())
    )

    data = raw_data.loc[raw_data["IDENT"].eq("1")].copy()

    if data.empty:
        raise RuntimeError("No standard AIH records found for the procedure.")

    if not (
        data["N_AIH"].str.fullmatch(r"\d{13}").fillna(False).all() and
        data["CNES" ].str.fullmatch(r"\d{7}" ).fillna(False).all()
    ):
        raise ValueError(
            "Invalid AIH or CNES identifier; review before consolidating records."
        )

    initial_count = len(data)

    competence = pd.to_datetime(
        data["ANO_CMPT"] + data["MES_CMPT"].str.zfill(2),

        format="%Y%m" ,
        errors="raise",
    )

    if competence.isna().any() or data[["source_file", "file_last_modified"]].isna().any().any():
        raise ValueError("Missing processing month or source provenance.")

    data["reference_month"] = competence.dt.strftime("%Y-%m").astype("string")

    data = data.sort_values(
        [
            "reference_month"   ,
            "file_last_modified",
            "source_file"       ,
        ]
    ).drop_duplicates(["N_AIH", "CNES"], keep="last")

    duplicate_count = initial_count - len(data)

    for source, target in [
        ("DT_INTER", "admission_date"),
        ("DT_SAIDA", "discharge_date"),
    ]:
        data[target] = pd.to_datetime(
            data[source],

            format="%Y%m%d",
            errors="raise" ,
        )

    if data[["admission_date", "discharge_date"]].isna().any().any():
        raise ValueError("Missing dates; review before selecting the study period.")

    if data["discharge_date"].lt(data["admission_date"]).any():
        raise ValueError("Discharge date precedes admission date.")

    within_period  = data["admission_date"].between(start_date, end_date)

    excluded_count = int((~within_period).sum())

    data.drop(index=data.index[~within_period], inplace=True)

    if data.empty:
        raise RuntimeError("No admissions found within the requested period.")

    return data, initial_count, duplicate_count, excluded_count


def _municipality_reference(municipalities):
    """Validate a locality snapshot and index it by the six-digit SIH prefix"""

    municipalities = pd.DataFrame(municipalities).copy()

    if "ibge_code" not in municipalities:
        municipalities["ibge_code"] = municipalities["id"].astype("string")

    municipalities["ibge_code"] = municipalities["ibge_code"].astype("string").str.strip()
    municipalities["nome"     ] = municipalities["nome"     ].astype("string").str.strip()

    if (
            municipalities.empty                                                       or
        not municipalities["ibge_code"].str.fullmatch(r"[0-9]{7}").fillna(False).all() or
            municipalities["nome"     ].isna(  )                                .any() or
            municipalities["nome"     ].eq  ("")                                .any()
    ):
        raise ValueError("Invalid IBGE municipality reference.")

    return municipalities.set_index(municipalities["ibge_code"].str[:6], verify_integrity=True)


def load_municipalities(root=None):
    """
    Reuse one IBGE locality snapshot for historical and current admission mapping

    Delete or replace the snapshot explicitly to update the reference. Geometry
    and transport-network coverage do not determine admission identification.
    """

    root = (
        Path(root)
        if   root is not None
        else Path(__file__).resolve().parents[1]
    )

    path = root / ".cache/ibge/municipalities.json"

    if path.is_file():
        raw            = path.read_bytes        ()
        municipalities = _municipality_reference(json.loads(raw))
    else:
        response = httpx.get(
            "https://servicodados.ibge.gov.br/api/v1/localidades/municipios", timeout=60,
        )

        response.raise_for_status()

        raw            = response.content
        municipalities = _municipality_reference(json.loads(raw))

        path.parent.mkdir(parents=True, exist_ok=True)

        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(raw)

            temporary_path = Path(temporary.name)

        try:
            temporary_path.replace(path)
        finally:
            temporary_path.unlink (missing_ok=True)

    municipalities.attrs.update(
        reference_file  =str   (path)            ,
        reference_sha256=sha256(raw ).hexdigest(),
    )

    return municipalities


def consolidate_admissions(data, municipalities=None):
    """Map IBGE municipality codes without excluding unmatched records"""

    municipalities = (
        load_municipalities()
        if   municipalities is None
        else _municipality_reference(municipalities)
    )

    data = data.copy()

    for source, target in [
        ("MUNIC_RES", "residence"),
        ("MUNIC_MOV", "procedure"),
    ]:
        data[f"{target}_ibge_code"   ] = data[source].map(municipalities["ibge_code"])
        data[f"{target}_municipality"] = data[source].map(municipalities["nome"     ])

    consolidated = data.rename(columns={
        "N_AIH"    : "aih_number"    ,
        "CNES"     : "cnes"          ,
        "PROC_REA" : "procedure_code",
    })

    return consolidated[
        [
            "aih_number"            ,
            "cnes"                  ,
            "procedure_code"        ,
            "admission_date"        ,
            "discharge_date"        ,
            "reference_month"       ,
            "residence_ibge_code"   ,
            "residence_municipality",
            "procedure_ibge_code"   ,
            "procedure_municipality",
        ]
    ].sort_values(
        [
            "admission_date",
            "aih_number"    ,
            "cnes"          ,
        ]
    ).reset_index(drop=True)
