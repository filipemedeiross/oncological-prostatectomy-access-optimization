"""Cataloging, caching, and extraction of DATASUS DBC files"""

import re
import os
import json

import pandas as pd

from pathlib            import Path
from time               import sleep
from hashlib            import sha256
from importlib.metadata import version
from functools          import partial
from tempfile           import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
from ftplib             import FTP, all_errors, error_perm


def _retry(operation):
    for attempt in range(3):
        try:
            return operation()
        except all_errors:
            if attempt == 2:
                raise

            sleep(2 ** attempt)


def list_files(
    host      ,
    directory ,
    states    ,
    start_date,
    end_date  ,
):
    """List source versions, retrieving timestamps and sizes together when possible"""

    def catalog():
        files = []

        with FTP(host, timeout=60) as ftp:
            ftp.login()
            ftp.cwd  (directory)

            try:
                entries = list(ftp.mlsd(facts=["modify", "size"]))
            except error_perm:
                entries = [
                    (name, {})
                    for name in ftp.nlst()
                ]

            ftp.voidcmd("TYPE I")
            for filename, facts in sorted(entries, key=lambda entry: entry[0]):
                match = re.fullmatch(
                    r"RD([A-Z]{2})(\d{2})(\d{2})\.dbc", filename, re.I
                )

                if not match:
                    continue

                state, year, month = match.groups()
                state              = state.upper ()
                reference_month    = f"20{year}-{month}"

                if state not in states or not start_date[:7] <= reference_month <= end_date[:7]:
                    continue

                last_modified = facts.get("modify") or ftp.sendcmd(f"MDTM {filename}").split()[1]
                file_size     = (
                    int(facts["size"])
                    if   "size" in facts
                    else ftp.size(filename)
                )

                if file_size is None or file_size <= 0 or not re.fullmatch(r"\d{14}(\.\d+)?", last_modified):
                    raise ValueError(f"Invalid FTP source metadata: {filename}.")

                files.append((filename, state, reference_month, last_modified, file_size))

        return files

    return _retry(catalog)


def extract_file(
    item     ,
    cache    ,
    host     ,
    directory,
    procedure,
    columns  ,

    *,

    filtered_cache=None,
    keep_dbc      =True,
):
    """
    Reuse DBCs; optionally persist only procedure-filtered Parquet records.

    Existing callers retain their DBC cache. Historical extraction can set
    keep_dbc=False and filtered_cache to avoid retaining full source downloads.
    """

    procedures = (
        (procedure,)
        if   isinstance(procedure, str)
        else tuple(sorted(set(procedure)))
    )

    if not procedures or any(not isinstance(p, str) or not re.fullmatch(r"\d{10}", p) for p in procedures):
        raise ValueError("Procedures must be ten-digit SIGTAP strings.")

    cache = Path(cache)
    dbc   = cache / f"{Path(item.filename).stem}_{item.last_modified}.dbc"

    cached = None
    if filtered_cache is not None:
        filtered_cache = Path(filtered_cache)

        filtered_cache.mkdir(parents=True, exist_ok=True)

        selection = json.dumps(
            {
                "version" : 1,

                "procedures" : sorted(procedures),
                "columns"    : list  (columns   ),
                "software"   : {
                    name : version(name)
                    for name in ("pysus", "pyreaddbc", "pyarrow")
                },
            },

            sort_keys=True,
        )

        signature = sha256(selection.encode()).hexdigest()[:16]
        cached    = filtered_cache / (
            f"{Path(item.filename).stem}_{item.last_modified}_{item.file_size}_{signature}.parquet"
        )

        if cached.is_file():
            return pd.read_parquet(cached).astype("string")

    temporary_root = filtered_cache if filtered_cache is not None else cache

    temporary_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYSUS_CACHEPATH", str(temporary_root.parent / "pysus"))

    from pyreaddbc             import dbc2dbf
    from pysus.data.dbf_reader import read_dbf_filtered

    with TemporaryDirectory(dir=temporary_root) as temporary_directory:
        temporary = Path(temporary_directory)

        if not dbc.is_file() or dbc.stat().st_size != item.file_size:
            downloaded = temporary / item.filename

            def download():
                with FTP(host, timeout=60) as ftp:
                    ftp.login()
                    ftp.cwd  (directory)

                    with downloaded.open("wb") as destination:
                        ftp.retrbinary(f"RETR {item.filename}", destination.write, blocksize=262144)

                if downloaded.stat().st_size != item.file_size:
                    raise OSError(f"Incomplete download: {item.filename}.")

            _retry(download)

            if keep_dbc:
                cache.mkdir(parents=True, exist_ok=True)

                downloaded.replace(dbc)
            else:
                dbc = downloaded

        dbf = temporary / "sih.dbf"
        dbc2dbf(str(dbc), str(dbf))

        data = read_dbf_filtered(
            dbf       ,
            "PROC_REA",

            list(procedures),

            columns     =columns,
            prefix_match=False  ,
        ).astype("string").assign(
            source_file       =item.filename     ,
            file_last_modified=item.last_modified,
        ).astype("string")

        if cached is not None:
            parquet = temporary / "filtered.parquet"

            data   .to_parquet(parquet, index=False)
            parquet.replace   (cached )

    return data


def extract_files(
    catalog  ,
    cache    ,
    host     ,
    directory,
    procedure,
    columns  ,

    *,

    filtered_cache=None,
    keep_dbc      =True,
    workers       =4   ,
):
    """Extract source files in catalog order with configurable filtered caching"""

    if catalog.empty or not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("A nonempty catalog and a positive integer worker count are required.")

    extract = partial(
        extract_file,

        cache         =cache         ,
        host          =host          ,
        directory     =directory     ,
        procedure     =procedure     ,
        columns       =columns       ,
        filtered_cache=filtered_cache,
        keep_dbc      =keep_dbc      ,
    )

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, data in enumerate(executor.map(extract, catalog.itertuples(index=False)), 1):
            results.append(data)

            if index % 27 == 0 or index == len(catalog):
                print(f"{index}/{len(catalog)} files processed", flush=True)

    return pd.concat(results, ignore_index=True).astype("string")
