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
    Hospitalizations from **01/01/2026 to 06/30/2026**, Brazil. One row per regular AIH and healthcare establishment, including place of residence, place of procedure and dates.

    - **Selection:** main procedure performed `PROC_REA = 0416010121` (oncology prostatectomy) and `IDENT = 1`;

    - **Served demand:** count by municipality of residence (`MUNIC_RES`);

    - **Observed supply:** count by municipality of the hospital (`MUNIC_MOV`);

    - **Reference date:** `DT_INTER`, the hospital admission date;

    - **Unit of analysis:** AIH, not unique patients;

    - **Outputs:** `data/oncology_prostatectomy_2026_first_half.csv` and the HTML report `reports/sih_data_processing.html`.

    > Note 01: Raw files are stored in `.cache/sih_prostatectomy` at the project root.
    """)
    return


@app.cell
def _():
    import os
    import re
    import httpx

    import pandas as pd

    from pathlib               import Path
    from time                  import sleep
    from pyreaddbc             import dbc2dbf
    from datetime              import datetime
    from pysus.data.dbf_reader import read_dbf_filtered

    from ftplib             import FTP, all_errors
    from concurrent.futures import ThreadPoolExecutor
    from tempfile           import TemporaryDirectory

    return (
        FTP,
        Path,
        TemporaryDirectory,
        ThreadPoolExecutor,
        all_errors,
        datetime,
        dbc2dbf,
        httpx,
        os,
        pd,
        re,
        read_dbf_filtered,
        sleep,
    )


@app.cell
def _(Path, os):
    ROOT = next(
        p
        for p in (Path.cwd(), *Path.cwd().parents)
        if (p / "notebooks/").is_dir()
    )

    os.environ["PYSUS_CACHEPATH"] = str(ROOT / ".cache/pysus")
    return (ROOT,)


@app.cell
def _(ROOT):
    COMPETENCIA_LIMITE = "2026-09"
    INICIO             = "2026-01-01"
    FIM                = "2026-06-30"

    PROCEDIMENTO = "0416010121"
    UFS          = "AC AL AM AP BA CE DF ES GO MA MG MS MT PA PB PE PI PR RJ RN RO RR RS SC SE SP TO".split()

    HOST      = "ftp.datasus.gov.br"
    DIRETORIO = "/dissemin/publicos/SIHSUS/200801_/Dados"

    CACHE = ROOT / ".cache/sih_prostatectomy"
    CSV   = ROOT / "data/oncology_prostatectomy_2026_first_half.csv"


    CACHE.       mkdir(parents=True, exist_ok=True)
    CSV  .parent.mkdir(parents=True, exist_ok=True)
    return (
        CACHE,
        COMPETENCIA_LIMITE,
        CSV,
        DIRETORIO,
        FIM,
        HOST,
        INICIO,
        PROCEDIMENTO,
        UFS,
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
def _(COMPETENCIA_LIMITE, DIRETORIO, FTP, HOST, INICIO, UFS, re):
    arquivos = []

    with FTP(HOST, timeout=60) as ftp:
        ftp.login()
        ftp.cwd  (DIRETORIO)

        nomes = sorted(ftp.nlst())

        ftp.voidcmd("TYPE I")
        for nome in nomes:
            match = re.fullmatch(
                r"RD([A-Z]{2})(\d{2})(\d{2})\.dbc", nome, re.I
            )

            if not match:
                continue

            uf, ano, mes = match.groups()
            competencia  = f"20{ano}-{mes}"

            if uf in UFS and INICIO[:7] <= competencia <= COMPETENCIA_LIMITE:
                atualizado = ftp.sendcmd(f"MDTM {nome}").split()[1]

                arquivos.append(
                    (
                        nome          ,
                        uf            ,
                        competencia   ,
                        atualizado    ,
                        ftp.size(nome),
                    )
                )
    return (arquivos,)


@app.cell
def _(FIM, INICIO, UFS, arquivos, pd):
    initial_catalogo = pd.DataFrame(
        arquivos,

        columns=[
            "arquivo"    ,
            "uf"         ,
            "competencia",
            "atualizado" ,
            "tamanho"    ,
        ]
    )

    catalogo = initial_catalogo.sort_values(
        [
            "competencia",
            "uf"         ,
        ]
    ).reset_index(drop=True)


    semestre  = pd.period_range(
        INICIO,
        FIM   ,

        freq="M",
    ).astype(str)

    cobertura = pd.crosstab(
        catalogo["competencia"],
        catalogo["uf"         ],
    ).reindex(columns=UFS, fill_value=0)

    faltantes = cobertura.reindex(
        semestre, fill_value=0
    ).eq(0).stack()


    if faltantes.any():
        raise RuntimeError(f"UF/competência ausente no semestre: {faltantes[faltantes].index.tolist()}")
    return catalogo, cobertura


@app.cell
def _(catalogo, datetime, mo):
    mo.md(f"""
    Resumo da consulta — SIH/SUS

    | Indicador | Valor |
    |:---|---:|
    | Arquivos encontrados | **{len(catalogo):,}** |
    | Última competência disponível | **{catalogo.competencia.max()}** |
    | Consulta realizada em | {datetime.now():%d/%m/%Y às %H:%M} |
    """)
    return


@app.cell
def _(cobertura, mo, pd):
    _resumo_cobertura = pd.DataFrame({
        "arquivos"     : cobertura.sum  (axis=1),
        "ufs_ausentes" : cobertura.apply(
            lambda linha: ", ".join(linha.index[linha.eq(0)]), axis=1
        ),
    }).rename_axis("competencia").reset_index()

    mo.ui.table(
        _resumo_cobertura,

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
    COLUNAS = [
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
    return (COLUNAS,)


@app.cell
def _(
    CACHE,
    COLUNAS,
    DIRETORIO,
    FTP,
    HOST,
    PROCEDIMENTO,
    Path,
    TemporaryDirectory,
    all_errors,
    dbc2dbf,
    read_dbf_filtered,
    sleep,
):
    def extrair(item):
        dbc = CACHE / f"{Path(item.arquivo).stem}_{item.atualizado}.dbc"

        if not dbc.exists() or dbc.stat().st_size != item.tamanho:
            for tentativa in range(3):
                try:
                    with FTP(HOST, timeout=60) as ftp:
                        ftp.login()
                        ftp.cwd  (DIRETORIO)

                        parcial = dbc.with_suffix(".part")

                        with parcial.open("wb") as destino:
                            ftp.retrbinary(
                                f"RETR {item.arquivo}",
                                destino.write         ,

                                blocksize=262144,
                            )

                    if parcial.stat().st_size != item.tamanho:
                        raise IOError(f"Download incompleto: {item.arquivo}")

                    parcial.replace(dbc)

                    break
                except all_errors as erro:
                    if tentativa == 2:
                        raise RuntimeError(f"Falha ao baixar {item.arquivo}") from erro

                    sleep(2 ** tentativa)

        with TemporaryDirectory(dir=CACHE) as temporario:
            dbf = Path(temporario) / "sih.dbf"

            dbc2dbf(str(dbc), str(dbf))

            parte = read_dbf_filtered(
                dbf       ,
                "PROC_REA",

                [
                    PROCEDIMENTO
                ],

                columns     =COLUNAS,
                prefix_match=False  ,
            )

        return parte.assign(
            arquivo_origem     =item.arquivo   ,
            atualizacao_arquivo=item.atualizado,
        )

    return (extrair,)


@app.cell
def _(ThreadPoolExecutor, catalogo, extrair, mo, pd):
    partes = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, parte in enumerate(
            executor.map(
                extrair                         ,
                catalogo.itertuples(index=False),
            ),
            1,
        ):
            partes.append(parte)

            if i % 27 == 0 or i == len(catalogo):
                print(f'{i}/{len(catalogo)} arquivos processados')


    brutos = pd.concat(partes, ignore_index=True).astype('string')


    with mo.redirect_stdout():
        print(f'{len(brutos):,} registros do procedimento antes do filtro por internação.')
    return (brutos,)


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
def _(brutos):
    initial_dados = brutos.loc[brutos["IDENT"].eq("1")].copy()

    if initial_dados.empty:
        raise RuntimeError("Nenhuma AIH normal encontrada para o procedimento.")

    if not (
        initial_dados["N_AIH"].str.fullmatch(r"\d{13}").fillna(False).all() and
        initial_dados["CNES" ].str.fullmatch(r"\d{7}" ).fillna(False).all()
    ):
        raise ValueError("AIH ou CNES inválido; revisar antes de consolidar registros.")
    return (initial_dados,)


@app.cell
def _(initial_dados):
    n_antes = len(initial_dados)

    initial_dados['competencia'] = (
        initial_dados['ANO_CMPT'] + '-' + initial_dados['MES_CMPT'].str.zfill(2)
    )

    dados = initial_dados.sort_values(
        [
            'competencia'        ,
            'atualizacao_arquivo',
            'arquivo_origem'     ,
        ]
    ).drop_duplicates(['N_AIH', 'CNES'], keep='last')

    n_repetidos = n_antes - len(dados)
    return dados, n_antes, n_repetidos


@app.cell
def _(FIM, INICIO, brutos, dados, mo, n_antes, n_repetidos, pd):
    for _origem, _destino in [
        ('DT_INTER', 'data_internacao'),
        ('DT_SAIDA', 'data_saida'     ),
    ]:
        dados[_destino] = pd.to_datetime(dados[_origem], format='%Y%m%d', errors='raise')

    if dados[['data_internacao', 'data_saida']].isna().any().any():
        raise ValueError('Data ausente; revisar antes de selecionar o semestre.')

    if dados['data_saida'].lt(dados['data_internacao']).any():
        raise ValueError('Saída anterior à internação.')

    no_semestre = dados['data_internacao'].between(INICIO, FIM)

    with mo.redirect_stdout():
        print(f'AIHs não normais = {len(brutos) - n_antes}')
        print(f'Reapresentações  = {n_repetidos          }')
        print(f'Fora do semestre = {(~no_semestre).sum() }')
    return (no_semestre,)


@app.cell
def _(dados, no_semestre):
    dados.drop(index=dados.index[~no_semestre], inplace=True)

    dados.head()

    if dados.empty:
        raise RuntimeError('Nenhuma internação no semestre solicitado.')
    return


@app.cell
def _(dados, httpx, pd):
    resposta = httpx.get(
        'https://servicodados.ibge.gov.br/api/v1/localidades/municipios', timeout=60
    )

    resposta.raise_for_status()


    municipios = pd.DataFrame(
        resposta.json()
    )[['id', 'nome']]

    municipios['codigo_ibge'] = municipios['id'].astype('string')
    municipios                = municipios.set_index   (
        municipios['codigo_ibge'].str[:6], verify_integrity=True
    )

    for _origem, _destino in [
        ('MUNIC_RES', 'residencia'),
        ('MUNIC_MOV', 'realizacao'),
    ]:
        dados[f'codigo_ibge_{_destino}'] = dados[_origem].map(municipios['codigo_ibge'])
        dados[f'municipio_{_destino}'  ] = dados[_origem].map(municipios['nome'       ])


    consolidated = dados.rename(
        columns={
            'N_AIH'    : 'numero_aih'         ,
            'CNES'     : 'cnes'               ,
            'PROC_REA' : 'codigo_procedimento',
        }
    )

    consolidated = consolidated[
        [
            'numero_aih'            ,
            'cnes'                  ,
            'codigo_procedimento'   ,
            'data_internacao'       ,
            'data_saida'            ,
            'competencia'           ,
            'codigo_ibge_residencia',
            'municipio_residencia'  ,
            'codigo_ibge_realizacao',
            'municipio_realizacao'  ,
        ]
    ].sort_values(['data_internacao', 'numero_aih', 'cnes']).reset_index(drop=True)


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
    Dates are stored in ISO format (`YYYY-MM-DD`), using UTF-8 encoding with BOM, and codes are preserved as text in the DataFrame. When reopening the CSV, use `dtype="string"` to preserve leading zeros in codes.
    """)
    return


@app.cell
def _(CSV, consolidated, mo):
    temporario_csv = CSV.with_suffix('.tmp')

    consolidated.to_csv(
        temporario_csv,

        index      =False      ,
        encoding   ='utf-8-sig',
        date_format='%Y-%m-%d' ,
    )

    temporario_csv.replace(CSV)


    with mo.redirect_stdout():
        print(f'{len(consolidated):,} internações salvas em {CSV}')
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
    pares = consolidated.dropna(subset=['codigo_ibge_residencia', 'codigo_ibge_realizacao'])

    fora_municipio = pares['codigo_ibge_residencia']        .ne(pares['codigo_ibge_realizacao']        )
    fora_uf        = pares['codigo_ibge_residencia'].str[:2].ne(pares['codigo_ibge_realizacao'].str[:2])

    demanda = consolidated.groupby(['codigo_ibge_residencia', 'municipio_residencia']).size().rename('demanda')
    supply  = consolidated.groupby(['codigo_ibge_realizacao', 'municipio_realizacao']).size().rename('supply' )

    mensal  = consolidated.groupby(consolidated['data_internacao'].dt.to_period('M')).size()
    return demanda, fora_municipio, fora_uf, mensal, pares, supply


@app.cell
def _(consolidated, demanda, mo, pares, pd, supply):
    _resumo = pd.Series(
        {
            'Internações (AIHs)' : len(consolidated),

            'Municípios de residência' : len(demanda),
            'Municípios de realização' : len(supply ),
            'Hospitais (CNES)'         : consolidated['cnes'].nunique(),

            'Residência sem correspondência IBGE' : consolidated['codigo_ibge_residencia'].isna().sum(),
            'Realização sem correspondência IBGE' : consolidated['codigo_ibge_realizacao'].isna().sum(),

            'Pares de municípios identificados' : len(pares),
        },

        name='valor'
    ).rename_axis('indicador').reset_index()


    mo.ui.table(
        _resumo,

        selection =None ,
        pagination=False,
        label     ="Hospitalization and municipality counts",
    )
    return


@app.cell
def _(consolidated, fora_municipio, fora_uf, mo, pd):
    _indicadores = pd.Series(
        {
            'Atendimento fora do município (%)' : 100 * fora_municipio.mean(),
            'Atendimento fora da UF (%)'        : 100 * fora_uf       .mean(),

            'Permanência mediana (dias)' : (consolidated['data_saida'] - consolidated['data_internacao']).dt.days.median(),
        },

        name='valor'
    ).round(2).rename_axis('indicador').reset_index()


    mo.ui.table(
        _indicadores,

        selection =None ,
        pagination=False,
        label     ="Travel and length of stay",
    )
    return


@app.cell
def _(FIM, INICIO, mensal, mo, pd):
    _mensal = mensal.reindex(
        pd.period_range(
            INICIO,
            FIM   ,

            freq="M",
        ),

        fill_value=0,
    ).rename_axis("mes_internacao").to_frame("internacoes").reset_index()


    mo.ui.table(
        _mensal,

        selection =None ,
        pagination=False,
        label     ="Hospitalizations by admission month",
    )
    return


@app.cell
def _(demanda, mo):
    mo.ui.table(
        demanda.nlargest(10).reset_index(),

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
