"""Cataloging, caching, and extraction of DATASUS DBC files"""

import re

import pandas as pd

from pathlib            import Path
from time               import sleep
from functools          import partial
from ftplib             import FTP, all_errors
from tempfile           import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

from pyreaddbc             import dbc2dbf
from pysus.data.dbf_reader import read_dbf_filtered


def list_files(
    host      ,
    directory ,
    states    ,
    start_date,
    end_date  ,
):
    files = []

    with FTP(host, timeout=60) as ftp:
        ftp.login()
        ftp.cwd  (directory)

        filenames = sorted(ftp.nlst())

        ftp.voidcmd("TYPE I")
        for filename in filenames:
            match = re.fullmatch(
                r"RD([A-Z]{2})(\d{2})(\d{2})\.dbc", filename, re.I
            )

            if not match:
                continue

            state, year, month = match.groups()
            state              = state.upper ()
            reference_month    = f"20{year}-{month}"

            if state in states and start_date[:7] <= reference_month <= end_date:
                last_modified = ftp.sendcmd(f"MDTM {filename}").split()[1]

                files.append(
                    (
                        filename          ,
                        state             ,
                        reference_month   ,
                        last_modified     ,
                        ftp.size(filename),
                    )
                )

    return files


def extract_file(
    item     ,
    cache    ,
    host     ,
    directory,
    procedure,
    columns  ,
):
    """Reuse a valid DBC file or download it; filter before materializing the DBF"""

    cache = Path(cache)
    dbc   = cache / f"{Path(item.filename).stem}_{item.last_modified}.dbc"

    if not dbc.exists() or dbc.stat().st_size != item.file_size:
        for attempt in range(3):
            try:
                with FTP(host, timeout=60) as ftp:
                    ftp.login()
                    ftp.cwd  (directory)

                    partial_file = dbc.with_suffix(".part")

                    with partial_file.open("wb") as destination:
                        ftp.retrbinary(
                            f"RETR {item.filename}",
                            destination.write      ,

                            blocksize=262144,
                        )

                if partial_file.stat().st_size != item.file_size:
                    raise IOError(f"Incomplete download: {item.filename}")

                partial_file.replace(dbc)

                break

            except all_errors as error:
                if attempt == 2:
                    raise RuntimeError(f"Failed to download {item.filename}") from error

                sleep(2 ** attempt)

    with TemporaryDirectory(dir=cache) as temporary_directory:
        dbf = Path(temporary_directory) / "sih.dbf"

        dbc2dbf(str(dbc), str(dbf))

        data = read_dbf_filtered(
            dbf       ,
            "PROC_REA",

            [
                procedure
            ],

            columns     =columns,
            prefix_match=False  ,
        )

    return data.assign(
        source_file       =item.filename     ,
        file_last_modified=item.last_modified,
    )


def extract_files(
    catalog  ,
    cache    ,
    host     ,
    directory,
    procedure,
    columns  ,
):
    """Extract catalog files using four workers while preserving their order"""

    Path(cache).mkdir(parents=True, exist_ok=True)

    extract = partial(
        extract_file,

        cache    =cache    ,
        host     =host     ,
        directory=directory,
        procedure=procedure,
        columns  =columns  ,
    )

    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, data in enumerate(
            executor.map(
                extract, catalog.itertuples(index=False)
            ),
            1,
        ):
            results.append(data)

            if index % 27 == 0 or index == len(catalog):
                print(f"{index}/{len(catalog)} files processed")

    return pd.concat(results, ignore_index=True).astype("string")
