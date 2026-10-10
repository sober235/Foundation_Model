#!/usr/bin/env python
"""Merge the shards of a sharded evaluation (scripts/aur_eval.py --shard K/N) into one report.

    PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/aur_eval_merge.py --shards <S0> <S1> <S2> \
        --samples <samples_1mm_v1.json> --split test [--val-patients ...] [--u-threshold SOURCE=THR ...] --out <NEW_DIR>

The union of the shards must be exactly the rows the unsharded run would evaluate (same split, sources and limit, all
from one checkpoint); the reference detector, the aggregate and the report are then built once, as aur_eval.py does."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.aur import eval as E  # noqa: E402
from anatobind.aur.ssl.checkpoint import code_sha  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aur_eval import select_rows  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=Path, nargs="+", required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--val-patients", type=Path, default=None)
    ap.add_argument("--u-threshold", action="append", default=[], metavar="SOURCE=THR")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists")
    infos = [json.loads((d / "shard.json").read_text()) for d in a.shards]
    if len({i["checkpoint_sha256"] for i in infos}) != 1 or len({i["split"] for i in infos}) != 1:
        raise ValueError("the shards come from different checkpoints or splits")
    results = [c for d in a.shards for c in json.loads((d / "cases.json").read_text())]
    expected = select_rows(json.loads(a.samples.read_text()), a.split, a.val_patients, infos[0]["sources"], infos[0]["limit"])
    key = lambda r: (r["source"], r["case"], r["sequence"])
    got = sorted(key(r) for r in results)
    if got != sorted(key(r) for r in expected):
        raise ValueError(f"the shards hold {len(got)} rows, the split {len(expected)}: missing or duplicated rows")
    order = {key(r): i for i, r in enumerate(expected)}
    results.sort(key=lambda r: order[key(r)])
    fixed = {s: float(t) for s, t in (item.split("=") for item in a.u_threshold)}
    reference = {}
    if not a.no_reference:
        for source in sorted({r["source"] for r in expected if r["source"] in E.REFERENCE}):
            cases = sorted({r["case"] for r in expected if r["source"] == source and r.get("u_supervised")})
            if cases:
                reference[source] = E.reference_summary(source, E.reference_scans(source, cases))
                print(f"reference {source}: {reference[source]}", flush=True)
    agg = E.aggregate(results, reference, fixed_thresholds=fixed)
    agg["run"] = {**{k: infos[0][k] for k in ("checkpoint", "checkpoint_sha256", "split", "samples", "samples_sha256", "crop", "sources", "limit",
                                               "instance_threshold", "stage", "e2e")},
                  "n_rows": len(results), "n_shards": len(infos), "shards": [str(d) for d in a.shards], "fixed_thresholds": fixed,
                  "code_sha_shards": sorted({i["code_sha"] for i in infos}), "code_sha_merge": code_sha(), "seconds_shards": [i["seconds"] for i in infos]}
    E.write_report(a.out, agg, results)
    (a.out / "reference.json").write_text(json.dumps(reference, indent=1, default=str))
    print(E.markdown(agg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
