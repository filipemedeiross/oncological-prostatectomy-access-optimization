import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ONCOLOGY PROSTATECTOMY — DATA PROCESSING
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hospitalizations from **01/01/2026 to 06/30/2026**, Brazil. One row per regular AIH and healthcare establishment, including place of residence, place of procedure, and dates.

    - **Selection:** main procedure performed `PROC_REA = 0416010121` (oncology prostatectomy) and `IDENT = 1`;
    - **Served demand:** count by municipality of residence (`MUNIC_RES`);
    - **Observed supply:** count by municipality of the hospital (`MUNIC_MOV`);
    - **Reference date:** `DT_INTER`, the hospital admission date;
    - **Unit of analysis:** AIH, not unique patients;
    - **Output:** `data/oncology_prostatectomy_2026_first_half.csv`.

    > Note: Raw files are stored in `.cache/sih_prostatectomy` at the project root.
    """)
    return


@app.cell
def _():
    import os
    import sys

    import pandas as pd

    from pathlib  import Path
    from datetime import datetime

    return Path, datetime, pd, sys


@app.cell
def _(Path, sys):
    ROOT = next(
        path
        for path in (Path.cwd(), *Path.cwd().parents)
        if (path / "notebooks").is_dir()
    )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return (ROOT,)


@app.cell
def _():
    from src.datasus import extract_files         , list_files
    from src.sih     import consolidate_admissions, prepare_catalog, process_admissions

    return (
        consolidate_admissions,
        extract_files,
        list_files,
        prepare_catalog,
        process_admissions,
    )


@app.cell
def _(ROOT):
    PROCESSING_PERIOD_CUTOFF = "2026-09"
    START_DATE               = "2026-01-01"
    END_DATE                 = "2026-06-30"

    PROCEDURE_CODE = "0416010121"

    STATES = "AC AL AM AP BA CE DF ES GO MA MG MS MT PA PB PE PI PR RJ RN RO RR RS SC SE SP TO".split()

    FTP_HOST      = "ftp.datasus.gov.br"
    FTP_DIRECTORY = "/dissemin/publicos/SIHSUS/200801_/Dados"

    CACHE_DIR = ROOT / ".cache/sih_prostatectomy"
    CSV_PATH  = ROOT / "data/oncology_prostatectomy_2026_first_half.csv"

    CACHE_DIR       .mkdir(parents=True, exist_ok=True)
    CSV_PATH .parent.mkdir(parents=True, exist_ok=True)
    return (
        CACHE_DIR,
        CSV_PATH,
        END_DATE,
        FTP_DIRECTORY,
        FTP_HOST,
        PROCEDURE_CODE,
        PROCESSING_PERIOD_CUTOFF,
        START_DATE,
        STATES,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## AVAILABLE FILES AND COVERAGE
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Availability is checked against the official FTP repository, including for Brazilian states (UFs) with no occurrences of the procedure. Execution is interrupted if any state/processing period within the semester is missing or if a download fails. Later processing periods may have partial coverage, as indicated in the table.
    """)
    return


@app.cell
def _(
    END_DATE,
    FTP_DIRECTORY,
    FTP_HOST,
    PROCESSING_PERIOD_CUTOFF,
    START_DATE,
    STATES,
    list_files,
    prepare_catalog,
):
    files = list_files(
        FTP_HOST                ,
        FTP_DIRECTORY           ,
        STATES                  ,
        START_DATE              ,
        PROCESSING_PERIOD_CUTOFF,
    )

    catalog, coverage = prepare_catalog(files, STATES, START_DATE, END_DATE)
    return catalog, coverage


@app.cell
def _(catalog, datetime, mo):
    mo.md(f"""
    **SIH/SUS query summary**

    | Indicator | Value |
    |:---|---:|
    | Files found | **{len(catalog):,}** |
    | Latest processing period available | **{catalog.reference_month.max()}** |
    | Query performed on | {datetime.now():%Y-%m-%d %H:%M} |
    """)
    return


@app.cell
def _(coverage, mo, pd):
    _coverage_summary = pd.DataFrame(
        {
            "file_count"     : coverage.sum  (axis=1),
            "missing_states" : coverage.apply(
                lambda row: ", ".join(row.index[row.eq(0)]), axis=1
            ),
        }
    ).rename_axis("reference_month").reset_index()

    mo.ui.table(
        _coverage_summary,

        selection =None ,
        pagination=False,
        label     ="Coverage by processing period",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## EXTRACTION
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Four simultaneous downloads are performed. Each DBC file is identified by its last-updated timestamp on the FTP server, downloaded only once, and filtered before the columns are materialized. The temporary DBF file is discarded after the read operation is completed.
    """)
    return


@app.cell
def _():
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
    return (COLUMNS,)


@app.cell
def _(
    CACHE_DIR,
    COLUMNS,
    FTP_DIRECTORY,
    FTP_HOST,
    PROCEDURE_CODE,
    catalog,
    extract_files,
    mo,
):
    raw_data = extract_files(
        catalog       ,
        CACHE_DIR     ,
        FTP_HOST      ,
        FTP_DIRECTORY ,
        PROCEDURE_CODE,
        COLUMNS       ,
    )

    with mo.redirect_stdout():
        print(f"{len(raw_data):,} procedure records before admission filtering.")
    return (raw_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FILTERING AND MUNICIPALITIES
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Invalid dates or missing identifiers interrupt execution. The original six-digit municipality codes are used internally to obtain municipality names and seven-digit IBGE codes from the official reference table. Municipalities with no matching entry remain in the CSV with empty derived fields and are counted in the analysis.
    """)
    return


@app.cell
def _(END_DATE, START_DATE, mo, process_admissions, raw_data):
    admissions, _initial_count, _duplicate_count, _excluded_count = process_admissions(
        raw_data, START_DATE, END_DATE
    )

    with mo.redirect_stdout():
        print(f"Non-standard AIHs  = {len(raw_data) - _initial_count}")
        print(f"Resubmissions      = {_duplicate_count}")
        print(f"Outside the period = {_excluded_count }")
    return (admissions,)


@app.cell
def _(admissions):
    admissions.head()
    return


@app.cell
def _(admissions, consolidate_admissions):
    consolidated = consolidate_admissions(admissions)

    consolidated.head()
    return (consolidated,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SINGLE CSV FILE
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Dates are stored in ISO format (`YYYY-MM-DD`), using UTF-8 encoding with BOM, and codes are preserved as text in the DataFrame. When reopening the CSV, specify code columns as strings to preserve leading zeros.
    """)
    return


@app.cell
def _(CSV_PATH, consolidated, mo):
    _temporary_csv = CSV_PATH.with_suffix(".tmp")

    consolidated.to_csv(
        _temporary_csv,

        index      =False,
        encoding   ="utf-8-sig",
        date_format="%Y-%m-%d" ,
    )

    _temporary_csv.replace(CSV_PATH)

    with mo.redirect_stdout():
        print(f"{len(consolidated):,} admissions saved to {CSV_PATH}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DESCRIPTIVE ANALYSIS
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The tables show monthly volume by **admission date**, municipalities of residence and procedure, care provided outside the municipality of residence, and the distribution of procedure volume across municipalities. Travel percentages are calculated only from records for which both municipalities were successfully identified using IBGE data. Municipal-level statistics include only municipalities with at least one identified hospitalization.

    **Limitations for interpretation:** results are restricted to SUS and to the selected main procedure. Late records not yet published may change the figures for the semester, particularly for June; processing-period coverage is documented above. For the article, report the processing-period cutoff, extraction date, exclusions, and the version of the CSV used.
    """)
    return


@app.cell
def _(consolidated):
    matched_pairs = consolidated.dropna(
        subset=["residence_ibge_code", "procedure_ibge_code"]
    )

    out_of_municipality = matched_pairs["residence_ibge_code"]        .ne(matched_pairs["procedure_ibge_code"])
    out_of_state        = matched_pairs["residence_ibge_code"].str[:2].ne(matched_pairs["procedure_ibge_code"].str[:2])

    demand = consolidated.groupby(["residence_ibge_code", "residence_municipality"]).size().rename("demand")
    supply = consolidated.groupby(["procedure_ibge_code", "procedure_municipality"]).size().rename("supply")

    monthly_admissions = consolidated.groupby(
        consolidated["admission_date"].dt.to_period("M")
    ).size()
    return (
        demand,
        matched_pairs,
        monthly_admissions,
        out_of_municipality,
        out_of_state,
        supply,
    )


@app.cell
def _(consolidated, demand, matched_pairs, mo, pd, supply):
    _summary = pd.Series(
        {
            "Admissions (AIHs)"                       : len(consolidated),
            "Municipalities of residence"             : len(demand),
            "Municipalities performing the procedure" : len(supply),

            "Healthcare establishments (CNES)"      : consolidated["cnes"].nunique(),
            "Residence without IBGE match"          : consolidated["residence_ibge_code"].isna().sum(),
            "Procedure location without IBGE match" : consolidated["procedure_ibge_code"].isna().sum(),

            "Identified municipality pairs" : len(matched_pairs),
        },
        name="value",
    ).rename_axis("indicator").reset_index()

    mo.ui.table(
        _summary,

        selection =None ,
        pagination=False,
        label     ="Hospitalization and municipality counts",
    )
    return


@app.cell
def _(consolidated, mo, out_of_municipality, out_of_state, pd):
    _indicators = pd.Series(
        {
            "Outside municipality (%)"     : 100 * out_of_municipality.mean(),
            "Outside state (%)"            : 100 * out_of_state       .mean(),
            "Median length of stay (days)" : (
                consolidated["discharge_date"] - consolidated["admission_date"]
            ).dt.days.median(),
        },

        name="value",
    ).round(2).rename_axis("indicator").reset_index()

    mo.ui.table(
        _indicators,

        selection =None ,
        pagination=False,
        label     ="Travel and length of stay",
    )
    return


@app.cell
def _(END_DATE, START_DATE, mo, monthly_admissions, pd):
    _monthly = monthly_admissions.reindex(
        pd.period_range(
            START_DATE,
            END_DATE  ,

            freq="M",
        ),
        fill_value=0,
    ).rename_axis("admission_month").to_frame("admissions").reset_index()

    mo.ui.table(
        _monthly,

        selection =None ,
        pagination=False,
        label     ="Hospitalizations by admission month",
    )
    return


@app.cell
def _(demand, mo):
    mo.ui.table(
        demand.nlargest(10).reset_index(),

        selection =None ,
        pagination=False,
        label     ="Top 10 municipalities of residence",
    )
    return


@app.cell
def _(mo, supply):
    mo.ui.table(
        supply.nlargest(10).reset_index(),

        selection =None ,
        pagination=False,
        label     ="Top 10 municipalities performing the procedure",
    )
    return


if __name__ == "__main__":
    app.run()
