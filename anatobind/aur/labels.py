"""Label spaces of the AnatoBind brain model (spec 2026-10-08 N2, §4).

A: one entity per SynthSeg structure of anatobind.eval.lookup.BRAIN_ALL (32, in that tuple's order); entity map value
i + 1 stands for ENTITY_LABELS[i], 0 for none. R: the 13 sided host classes of the S4 label space
(anatobind.anatomy.labels.HOST_IDS, in that order) as host indices 0..12, plus NO_HOST = 13 for a lesion without a
parenchymal host within NEAR_MM; ventricles and CSF are landmarks, never hosts (V7 §5). S: six sequence types.
Everything derived from SynthSeg is a pseudo-label: NOT_EVIDENCE."""
import numpy as np

from anatobind.anatomy.labels import HOST_CLASS_OF, HOST_IDS, NAMES, SIDE_OF, to_student
from anatobind.eval.lookup import BRAIN_ALL

ENTITY_LABELS = tuple(int(l) for l in BRAIN_ALL)      # SynthSeg values; entity index = position in this tuple
N_ENTITIES = len(ENTITY_LABELS)                       # 32
ENTITY_INDEX = {label: i for i, label in enumerate(ENTITY_LABELS)}
N_HOSTS = len(HOST_IDS)                               # 13
NO_HOST = N_HOSTS                                     # host index 13
N_HOST_CLASSES = N_HOSTS + 1
HOST_NAMES = tuple(NAMES[c] for c in HOST_IDS) + ("no_host",)
HOST_TISSUE = tuple(HOST_CLASS_OF[c] for c in HOST_IDS)        # tissue family of each host index
HOST_SIDE = tuple(SIDE_OF[c] for c in HOST_IDS)                # "left" / "right" / None (brainstem)
TISSUES = tuple(dict.fromkeys(HOST_TISSUE))                    # the 7 tissue families, first-seen order
SEQ_TYPES = ("T1", "T1c", "T2", "FLAIR", "DWI", "ADC")
SEQ_INDEX = {s: i for i, s in enumerate(SEQ_TYPES)}
NEAR_MM = 10.0                                        # a lesion that overlaps no host takes the nearest one within this

_ENTITY_LUT = np.zeros(256, np.uint8)
for _i, _l in enumerate(ENTITY_LABELS):
    _ENTITY_LUT[_l] = _i + 1
_HOST_LUT = np.zeros(16, np.uint8)                    # S4 student id -> host index + 1 (ventricles 14 -> 0)
for _i, _c in enumerate(HOST_IDS):
    _HOST_LUT[_c] = _i + 1
ENTITY_HOST = tuple(int(_HOST_LUT[to_student(np.array([l]))[0]]) - 1 for l in ENTITY_LABELS)   # host index or -1


def entity_map(seg):
    """SynthSeg map -> uint8 entity map (0 none, i + 1 for ENTITY_LABELS[i])."""
    s = np.asarray(seg)
    if s.min() < 0 or s.max() > 255:
        raise ValueError(f"label values outside 0..255: {int(s.min())}..{int(s.max())}")
    return _ENTITY_LUT[s.astype(np.int64)]


def entity_to_synthseg(emap):
    """Entity map -> SynthSeg values (int16); 0 stays 0."""
    e = np.asarray(emap)
    if e.min() < 0 or e.max() > N_ENTITIES:
        raise ValueError(f"entity values outside 0..{N_ENTITIES}: {int(e.min())}..{int(e.max())}")
    lut = np.zeros(N_ENTITIES + 1, np.int16)
    lut[1:] = ENTITY_LABELS
    return lut[e.astype(np.int64)]


def host_map(seg):
    """SynthSeg map -> uint8 host map (0 none: background, ventricles, CSF, unlisted; i + 1 for host index i)."""
    return _HOST_LUT[to_student(seg).astype(np.int64)]


def contralateral(host):
    """Host index of the same tissue on the other side; None for the brainstem."""
    side = HOST_SIDE[host]
    if side is None:
        return None
    other = "right" if side == "left" else "left"
    return next(i for i in range(N_HOSTS) if HOST_TISSUE[i] == HOST_TISSUE[host] and HOST_SIDE[i] == other)
