#!/usr/bin/env python
# scripts/level_r_report.py
"""Level R pilot report (spec §8, §9): reader agreement, strata, reading time and the R7 gate, written as markdown with
the command and the raw summary attached.

  D=/data2/congcong/data/FM_data/derived/level_r
  python scripts/level_r_report.py --db $D/level_r.sqlite --pilot data/level_r/pilot_150.json \
      --out docs/verification/$(date +%F)/level_r_pilot.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.level_r_stats import summarise  # noqa: E402
from anatobind.level_r.registry import REGISTRY, load_registry  # noqa: E402
from anatobind.level_r.store import Store  # noqa: E402

FENCE = "`" * 3          # built at runtime so this file itself stays out of markdown fences


def build_summary(store, registry, lesion_ids, n_boot, seed):
    readers = sorted(r["reader_id"] for r in store.readers() if r["role"] == "reader")
    if len(readers) != 2:
        raise ValueError(f"need exactly two readers, have {readers}")
    a, b = (store.latest_labels(r) for r in readers)
    return {"readers": readers, **summarise(a, b, registry, lesion_ids, n_boot=n_boot, seed=seed)}


def _f(x):
    return "nan" if x is None or x != x else f"{x:.3f}"


def _table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def render_markdown(s, title, command):
    g = s["gate_r7"]
    lines = [f"# {title}", "",
             f"readers {s['readers']} · pairs {s['n_pairs']} · patients {s['n_patients']}", "",
             f"raw agreement {_f(s['raw'])} (95% patient-bootstrap CI {_f(s['raw_ci95'][0])}–{_f(s['raw_ci95'][1])}) · "
             f"Cohen κ {_f(s['kappa'])} · Gwet AC1 {_f(s['ac1'])} · set-valued agreement {_f(s['set_agreement'])}", "",
             f"GATE_R7: {'PASS' if g['single_host_endpoint_allowed'] else 'FAIL'} "
             f"(all CI low {_f(g['all_ci_low'])} vs 0.80 -> {g['pass_all']}; 0 mm band raw {_f(g['band0_raw'])} vs 0.70 -> {g['pass_band0']})",
             "", "## positive agreement per class", "",
             _table(["class", "n reader A", "n reader B", "positive agreement"],
                    [[c, v["n_x"], v["n_y"], _f(v["positive_agreement"])] for c, v in s["positive_agreement"].items()]),
             "", "## confusion (rows reader A, columns reader B)", "",
             _table([""] + s["confusion"]["categories"], [[c] + row for c, row in zip(s["confusion"]["categories"], s["confusion"]["counts"])]),
             "", "## strata", ""]
    for key in ("band", "stratum_geometry"):
        lines += [f"### {key}", "", _table([key, "n", "raw", "set agreement"],
                                          [[k, v["n"], _f(v["raw"]), _f(v["set_agreement"])] for k, v in s["strata"][key].items()]), ""]
    t3 = s["strata"]["slice_3mm"]
    lines += [f"slice_3mm volumes: n {t3['n']}, raw {_f(t3['raw'])}, set agreement {_f(t3['set_agreement'])}", "", "## reading time", ""]
    for name, t in s["time"].items():
        lines.append(f"- {name}: n {t['n']}, median {t['median_s']} s (IQR {t['q1_s']}–{t['q3_s']}), hours_for_1297 {t['hours_for_1297']}")
    lines += ["", "## command", "", FENCE, command, FENCE, "", "## summary (raw)", "", FENCE + "json", json.dumps(s, indent=1, default=list), FENCE, ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--pilot", type=Path, help="pilot json; restricts the report to its lesion_ids")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="Level R pilot agreement")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; reports are never overwritten")
    ids = json.loads(a.pilot.read_text(encoding="utf-8"))["lesion_ids"] if a.pilot else None
    s = build_summary(Store(a.db), load_registry(a.registry), ids, a.n_boot, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render_markdown(s, a.title, " ".join(sys.argv)), encoding="utf-8")
    print(f"pairs {s['n_pairs']} raw {s['raw']:.3f} CI {s['raw_ci95']} GATE_R7 {'PASS' if s['gate_r7']['single_host_endpoint_allowed'] else 'FAIL'} -> {a.out}")


if __name__ == "__main__":
    main()
