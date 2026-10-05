"""Fetch Roboflow annotation jobs and map every image to its labeler.

Writes ``reports/filenames_by_labeler.json`` as {labeler: [filename, ...]}.
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

from roboflow import Roboflow

WORKSPACE = "belns-workspace"
PROJECT = "defensible_space_lawal"
OUT = Path("reports/filenames_by_labeler.json")


def main() -> None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        for ln in Path(".env").read_text().splitlines():
            if ln.startswith("ROBOFLOW_API_KEY="):
                key = ln.split("=", 1)[1].strip()
                break
    rf = Roboflow(api_key=key)
    pj = rf.workspace(WORKSPACE).project(PROJECT)

    jobs = pj.get_annotation_jobs().get("jobs", [])
    print(f"{len(jobs)} jobs total")

    by_labeler: dict[str, set[str]] = defaultdict(set)
    for j in jobs:
        labeler = j.get("labeler") or "unknown"
        print(f"  job '{j['name'][:50]}' labeler={labeler} status={j.get('status')}")
        ids: list[str] = []
        for page in pj.search_all(annotation_job_id=j["id"], limit=100):
            ids.extend(x["id"] for x in page)
        print(f"    {len(ids)} images, fetching names...")
        for i, img_id in enumerate(ids):
            info = pj.image(img_id)
            by_labeler[labeler].add(info["name"])
            if (i + 1) % 100 == 0:
                print(f"      {i + 1}/{len(ids)}")
            time.sleep(0.03)

    serializable = {k: sorted(v) for k, v in by_labeler.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(serializable, indent=2))
    print(f"\nwrote {OUT}")
    for k, v in sorted(by_labeler.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:40s} {len(v)} images")


if __name__ == "__main__":
    main()
