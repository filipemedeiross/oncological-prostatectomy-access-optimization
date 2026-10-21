"""Coverage validation, SIH admission selection, and municipality identification"""

import httpx

import pandas as pd


def prepare_catalog(
    files     ,
    states    ,
    start_date,
    end_date  ,
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

    months   = pd.period_range(start_date, end_date, freq="M").astype(str)

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

    data["reference_month"] = (
        data["ANO_CMPT"] + "-" + data["MES_CMPT"].str.zfill(2)
    )

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


def consolidate_admissions(data):
    """Map IBGE municipality codes without excluding unmatched records"""

    response = httpx.get(
        "https://servicodados.ibge.gov.br/api/v1/localidades/municipios", timeout=60,
    )

    response.raise_for_status()

    municipalities = pd.DataFrame(
        response.json()
    )[["id", "nome"]]

    municipalities["ibge_code"] = municipalities["id"].astype("string")

    municipalities = municipalities.set_index(
        municipalities["ibge_code"].str[:6], verify_integrity=True,
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
