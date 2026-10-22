"""Download and validate the original May 2024 Saldanha Parquet from Zenodo"""

import sys
import httpx
import hashlib

from pathlib  import Path
from tempfile import NamedTemporaryFile


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)


from src.distances import validate_parquet, DEFAULT_PARQUET_PATH


SOURCE_URL   = "https://zenodo.org/records/11400243/files/dist_brasil.parquet?download=1"
OFFICIAL_MD5 = "dfd2d64bc045d0d0bb57d698dbec7ae1"


def validate_download(path):
    """Match Zenodo's published MD5, then inspect the nullable source measures"""

    with Path(path).open("rb") as stream:
        checksum = hashlib.file_digest(stream, "md5").hexdigest()

    if checksum != OFFICIAL_MD5:
        raise ValueError(f"Saldanha checksum mismatch: expected {OFFICIAL_MD5}, got {checksum}.")

    validate_parquet(path)


def download_saldanha(path=DEFAULT_PARQUET_PATH):
    """Reuse valid originals and replace the destination only after validation"""

    path = Path(path)

    if path.is_file():
        try:
            validate_download(path)
        except ValueError:
            print(f"Invalid cached file; downloading a replacement: {path}")
        else:
            print(f"Using validated Saldanha Parquet: {path}")

            return path

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = None

    try:
        with NamedTemporaryFile(
            dir   =path.parent   ,
            suffix=".parquet.tmp",
            delete=False         ,
        ) as stream:
            temporary = Path(stream.name)

            with httpx.stream("GET", SOURCE_URL, follow_redirects=True, timeout=120) as response:
                response.raise_for_status()

                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    stream.write(chunk)

        validate_download(temporary)

        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    print(f"Downloaded and validated Saldanha Parquet: {path}")

    return path


if __name__ == "__main__":
    download_saldanha()
