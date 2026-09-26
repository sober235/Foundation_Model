"""Pilot sample (spec R8 / §9): n lesions over the cells distance band x measured-geometry stratum, every non-empty
cell represented, no patient dominating. The pilot only estimates agreement and reading time (v2.6 §7.2); the
agreement gate is judged on the full set."""
from collections import Counter, defaultdict

import numpy as np


def cell_of(row):
    return f"{row['band']}|{row['stratum_geometry']}"


def allocate(cell_sizes, n, min_per_cell):
    """{cell: size} -> {cell: allocation}. Cells below min_per_cell are taken whole; cells whose proportional share
    is below min_per_cell get exactly min_per_cell; the remaining budget is split among the other cells in proportion
    to size, largest-remainder rounding."""
    total = sum(cell_sizes.values())
    fixed, flex = {}, {}
    for c, s in cell_sizes.items():
        if s < min_per_cell:
            fixed[c] = s
        elif n * s / total < min_per_cell:
            fixed[c] = min_per_cell
        else:
            flex[c] = s
    budget = n - sum(fixed.values())
    if budget < min_per_cell * len(flex):
        raise ValueError(f"budget {budget} after the fixed cells cannot give {len(flex)} cells {min_per_cell} each")
    if not flex:
        return fixed
    flex_total = sum(flex.values())
    raw = {c: budget * s / flex_total for c, s in flex.items()}
    alloc = {c: int(np.floor(v)) for c, v in raw.items()}
    for c in sorted(flex, key=lambda c: (-(raw[c] - alloc[c]), c))[: budget - sum(alloc.values())]:
        alloc[c] += 1
    for c, a in alloc.items():
        if a < min_per_cell or a > flex[c]:
            raise ValueError(f"cell {c}: allocation {a} outside [{min_per_cell}, {flex[c]}]")
    return {**fixed, **alloc}


def sample_pilot(registry, n=150, min_per_cell=8, max_per_patient=3, seed=0):
    cells = defaultdict(list)
    for r in registry:
        cells[cell_of(r)].append(r)
    alloc = allocate({c: len(v) for c, v in cells.items()}, n, min_per_cell)
    rng = np.random.default_rng(seed)
    per_patient, picked, leftovers = Counter(), [], {}
    for c in sorted(cells, key=lambda c: (len(cells[c]), c)):            # small cells draw first
        rows = sorted(cells[c], key=lambda r: r["lesion_id"])
        got, rest = [], []
        for r in (rows[int(i)] for i in rng.permutation(len(rows))):
            if len(got) < alloc[c] and per_patient[r["patient_id"]] < max_per_patient:
                got.append(r)
                per_patient[r["patient_id"]] += 1
            else:
                rest.append(r)
        picked += got
        leftovers[c] = rest
    short = n - len(picked)
    for c in sorted(cells, key=lambda c: (-len(cells[c]), c)):           # refill from the largest cells
        for r in leftovers[c]:
            if short <= 0:
                break
            if per_patient[r["patient_id"]] < max_per_patient:
                picked.append(r)
                per_patient[r["patient_id"]] += 1
                short -= 1
    if len(picked) != n:
        raise ValueError(f"could only draw {len(picked)} of {n} lesions under the {max_per_patient}-per-patient cap")
    counts = Counter(cell_of(r) for r in picked)
    return {"n": n, "seed": seed, "min_per_cell": min_per_cell, "max_per_patient": max_per_patient,
            "lesion_ids": sorted(r["lesion_id"] for r in picked),
            "cells": {c: {"n": len(cells[c]), "alloc": alloc[c], "picked": counts.get(c, 0)} for c in sorted(cells)}}
