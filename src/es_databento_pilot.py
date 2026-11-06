"""Quote or download the fixed Databento March 2025 ES roll pilot."""

import argparse
from datetime import datetime, timezone
from hashlib import file_digest
from importlib.metadata import version
import json
import math
import os
from pathlib import Path

from .data_download import RAW

QUERY = {
    "dataset": "GLBX.MDP3",
    "symbols": ["ESH5", "ESM5"],
    "stype_in": "raw_symbol",
    "start": "2025-03-13T00:00:00Z",
    "end": "2025-03-19T00:00:00Z",
}
SCHEMAS = ("definition", "trades")


def quote_pilot(client):
    costs = {schema: client.metadata.get_cost(schema=schema, **QUERY) for schema in SCHEMAS}
    if any(not math.isfinite(cost) or cost < 0 for cost in costs.values()):
        raise ValueError("Databento returned an invalid quote")
    return {
        "request": QUERY,
        "estimated_usd_by_schema": costs,
        "estimated_total_usd": math.fsum(costs.values()),
    }


def download_pilot(client, quote, max_usd, out, sdk_version):
    if not math.isfinite(max_usd) or max_usd < 0 or quote["estimated_total_usd"] > max_usd:
        raise ValueError("The estimated cost exceeds the approved budget or the budget is invalid")
    # Reserve a fresh destination before any billable call. Leave successful
    # parts and the quote available for inspection if a later request fails.
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        **quote, "databento_version": sdk_version,
        "requested_at_utc": datetime.now(timezone.utc).isoformat(), "files": {},
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for schema in SCHEMAS:
        path = out / f"{schema}.dbn.zst"
        pending = out / f"{schema}.partial.dbn.zst"
        client.timeseries.get_range(schema=schema, path=pending, **QUERY)
        if not pending.stat().st_size:
            raise ValueError(f"Empty Databento file for {schema}")
        with pending.open("rb") as data:
            checksum = file_digest(data, "sha256").hexdigest()
        pending.replace(path)
        manifest["files"][schema] = {"name": path.name, "bytes": path.stat().st_size, "sha256": checksum}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-usd", type=float, help="Permit download up to this USD estimate")
    parser.add_argument("--out", type=Path, default=RAW / "databento" / "es-roll-pilot")
    args = parser.parse_args()
    if not os.environ.get("DATABENTO_API_KEY"):
        parser.error("Set DATABENTO_API_KEY in your local environment")
    try:
        import databento as db
    except ImportError:
        parser.error("Install the optional Databento client with python -m pip install databento")
    client = db.Historical()
    quote = quote_pilot(client)
    print(json.dumps(quote, indent=2), flush=True)
    if args.max_usd is not None:
        try:
            download_pilot(client, quote, args.max_usd, args.out, version("databento"))
        except (ValueError, FileExistsError) as exc:
            parser.error(str(exc))
        print(args.out)


if __name__ == "__main__":
    main()
