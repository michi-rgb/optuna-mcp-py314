"""Download the audited dependency wheels and verify their SHA-256 digests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import os
from pathlib import Path
import platform
import re
import sys
from tempfile import NamedTemporaryFile
from urllib.parse import unquote, urldefrag, urlsplit
from urllib.request import urlopen


PROJECT = Path(__file__).resolve().parents[1]
HOSTS = {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"}
TARGETS = {("win32", "amd64"): "windows-x64", ("linux", "x86_64"): "linux-x64"}


def wheel_records(target):
    manifest = PROJECT / f"wheels-{target}.txt"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        _, source = line.split(" @ ", 1)
        url, fragment = urldefrag(source)
        parsed = urlsplit(url)
        filename = unquote(parsed.path.rsplit("/", 1)[-1])
        if (
            parsed.scheme != "https"
            or parsed.netloc not in HOSTS
            or not re.fullmatch(r"[A-Za-z0-9_.+!-]+\.whl", filename)
            or not re.fullmatch(r"sha256=[a-f0-9]{64}", fragment)
        ):
            raise ValueError(f"Invalid wheel entry in {manifest.name}")
        yield filename, url, fragment.removeprefix("sha256=")


def verify_or_download(destination, record):
    filename, url, expected = record
    wheel = destination / filename
    if wheel.is_file():
        with wheel.open("rb") as existing:
            if file_digest(existing, "sha256").hexdigest() == expected:
                return

    with NamedTemporaryFile(dir=destination, suffix=".part", delete=False) as output:
        temporary = Path(output.name)
    try:
        digest = sha256()
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {filename}")
        os.replace(temporary, wheel)
        print(f"Verified {filename}", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS.values()))
    args = parser.parse_args()
    target = args.target or TARGETS.get((sys.platform, platform.machine().lower()))
    if target is None:
        parser.error("Supported targets are Windows x64 and Linux x64.")
    destination = PROJECT / "wheelhouse" / target
    records = list(wheel_records(target))
    if not records:
        raise ValueError("The wheel list is empty.")
    destination.mkdir(parents=True, exist_ok=True)
    expected_files = {filename for filename, _, _ in records}
    unexpected_files = {entry.name for entry in destination.glob("*.whl")} - expected_files
    if unexpected_files:
        raise ValueError(f"Unexpected wheels in {destination}: {sorted(unexpected_files)}")
    with ThreadPoolExecutor(max_workers=6) as executor:
        for _ in executor.map(lambda record: verify_or_download(destination, record), records):
            pass
    print(f"Verified {len(records)} dependency wheels in {destination}")


if __name__ == "__main__":
    main()
