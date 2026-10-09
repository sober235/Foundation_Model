"""The sample table: one row per (case, sequence) with its supervision flags and split (spec 2026-10-08 §3).

A row carries the image, the SynthSeg map on the same grid (A is always supervised), the source lesion map and the
label values that mean "lesion on this sequence" (U is supervised only where the lesion is visible, N3). Sources:
PDGM, BMSR, ISLES through anatobind.nnunet.brain_disease; SibBMS through anatobind.anatomy.sources (A and S only:
its lesions are not annotated on the template grid). Splits follow the S4 case table for PDGM / BMSR / SibBMS and
split_by_patient for ISLES (N12)."""
import json
from pathlib import Path

from anatobind.anatomy.sources import FM, check_split, sibbms_cases, split_by_patient
from anatobind.nnunet.brain_disease import anatomy_path, channel_path, label_path, list_cases, patient_of

# source -> {name of the sequence in the source: (sequence type, lesion values that are visible on it; () = A/S only)}
SEQUENCES = {
    "pdgm": {"T1": ("T1", ()), "T1c": ("T1c", (1, 4)), "T2": ("T2", ()), "FLAIR": ("FLAIR", (1, 2, 4))},
    "bmsr": {"T1pre": ("T1", ()), "T1post": ("T1c", (1,)), "FLAIR": ("FLAIR", ())},
    "isles": {"DWI": ("DWI", (1,)), "ADC": ("ADC", ())},
    "sibbms": {"T1w": ("T1", ()), "T2w": ("T2", ()), "FLAIR": ("FLAIR", ())},
}
ALL_LESION_VALUES = {"pdgm": (1, 2, 4), "bmsr": (1,), "isles": (1,), "sibbms": ()}     # every lesion voxel, any sequence
DISEASE_OF = {"pdgm": "glioma", "bmsr": "metastasis", "isles": "infarct"}
SOURCES = ("pdgm", "bmsr", "isles", "sibbms")


def _row(case, source, patient, seq_name, image, anatomy, lesion):
    seq_type, values = SEQUENCES[source][seq_name]
    return {"case": case, "source": source, "patient": patient, "sequence": seq_type, "source_sequence": seq_name,
            "image": str(image), "anatomy": str(anatomy), "lesion": None if lesion is None else str(lesion),
            "u_supervised": bool(values), "u_values": list(values), "a_ignore_values": list(ALL_LESION_VALUES[source])}


def disease_samples(source, root=FM):
    """Rows of PDGM / BMSR / ISLES: every sequence of every case."""
    d = DISEASE_OF[source]
    rows = []
    for case in list_cases(d, root):
        for seq_name in SEQUENCES[source]:
            rows.append(_row(case, source, patient_of(d, case), seq_name, channel_path(d, case, seq_name, root),
                             anatomy_path(d, case, root), label_path(d, case, root)))
    return rows


def sibbms_samples(root=FM):
    """Rows of SibBMS: the sequences present beside each session's FLAIR (T1w, T2w, FLAIR); no lesion map."""
    rows = []
    for case, r in sibbms_cases(root).items():
        flair = Path(r["flair"])
        stem = flair.name[:-len("_FLAIR.nii.gz")]
        for seq_name in SEQUENCES["sibbms"]:
            p = flair.parent / f"{stem}_{seq_name}.nii.gz"
            if p.is_file():
                rows.append(_row(case, "sibbms", r["patient"], seq_name, p, r["anatomy"], None))
    return rows


def all_samples(root=FM):
    rows = []
    for source in SOURCES:
        rows += sibbms_samples(root) if source == "sibbms" else disease_samples(source, root)
    return rows


def assign_splits(rows, s4_cases, seed=0):
    """Add "split" to every row. PDGM / BMSR / SibBMS cases take the split of the S4 case table (the same test
    patients as S4); ISLES cases are split by patient here. Every case must get a split; no patient may be in both."""
    split = {case: r["split"] for case, r in s4_cases.items()}
    isles = {r["case"]: {"source": "isles", "patient": r["patient"]} for r in rows if r["source"] == "isles"}
    if isles:
        split.update(split_by_patient(isles, seed=seed))
    out = []
    for r in rows:
        if r["case"] not in split:
            raise KeyError(f"{r['case']} ({r['source']}): no split for this case")
        out.append({**r, "split": split[r["case"]]})
    by_case = {r["case"]: r for r in out}
    check_split(by_case, {c: r["split"] for c, r in by_case.items()})
    return out


def counts(rows):
    """{source: {"train": (cases, rows, rows with U), "test": ...}} for the records."""
    out = {}
    for split in ("train", "test"):
        for source in SOURCES:
            sel = [r for r in rows if r["source"] == source and r["split"] == split]
            out.setdefault(source, {})[split] = (len({r["case"] for r in sel}), len(sel), sum(r["u_supervised"] for r in sel))
    return out


def write_samples(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; nothing is overwritten here, use a new name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))


def read_samples(path):
    return json.loads(Path(path).read_text())
