"""Fetch immutable research inputs and verify their SHA-256 digests."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
SPY_REVISION = "25eafd054ddb16d5471761312a0c852a672d6431"
SPY_URL = (
    "https://api.github.com/repos/Setooooo/etf-insight/contents/data/raw/SPY.csv"
    f"?ref={SPY_REVISION}"
)
SPY_SHA256 = "fc9bfeae3ad120f4535628aa17e4de697b24185ee1c010bb3f21934b680cbb6d"
ES_REVISION = "29efdfa21c5a5b2d7aa306397385cf116e011559"
ES_URL = (
    "https://huggingface.co/datasets/lynx1231/historical-futures-data-sample"
    f"/resolve/{ES_REVISION}/data/daily/CME/ES"
)
ES_SHA256 = {
    "ESH25": "bb3a6e0941f0b87f7e6f1c885fd33fe612b5e73da956c92bb8d6ba5020781192",
    "ESM25": "94b12627eeb2d9770fa3054391c48e4f0f9616dadd297ec2f2ad5a483168d422",
    "ESU25": "8d9997956b64eb8faea3e049aa0ae9929484f8f014259251232ac26e42206eab",
    "ESZ25": "0ababbaefffde691e05cd69a1df85070bbacb32036c41adffb9839a99d670204",
    "ESH26": "219ecbd9462dc52587265fcf6aad486268cc792535f9c5384f19b9e9f35547fc",
}


def check_digest(data, expected, source):
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {source}. Expected {expected}, got {actual}.")


def fetch(path, url, digest, github=False):
    if path.exists():
        check_digest(path.read_bytes(), digest, path)
    else:
        request = Request(url, headers={"User-Agent": "uncertainty-rl-research"})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        if github:
            payload = base64.b64decode(json.loads(payload)["content"])
        check_digest(payload, digest, url)
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as scratch:
            staged = Path(scratch) / path.name
            staged.write_bytes(payload)
            staged.replace(path)
    return path


def get_spy(raw=RAW):
    return fetch(raw / "SPY.csv", SPY_URL, SPY_SHA256, github=True)


def get_es(raw=RAW):
    return {
        symbol: fetch(raw / "es" / f"{symbol}.parquet", f"{ES_URL}/{symbol}.parquet", digest)
        for symbol, digest in ES_SHA256.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    args = parser.parse_args()
    print(get_spy(args.raw_dir))
    for path in get_es(args.raw_dir).values():
        print(path)


if __name__ == "__main__":
    main()
