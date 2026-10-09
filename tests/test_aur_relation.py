# tests/test_aur_relation.py
import torch

import anatobind.aur.relation as R
from anatobind.aur.labels import ENTITY_HOST, ENTITY_INDEX, HOST_NAMES, HOST_SIDE, N_ENTITIES, N_HOSTS, TISSUES


def test_host_grouping_of_entities():
    emb = torch.zeros(1, N_ENTITIES, 4)
    for i, h in enumerate(ENTITY_HOST):
        emb[0, i] = h if h >= 0 else 99
    hosts = R.host_from_entities(emb)
    assert hosts.shape == (1, N_HOSTS, 4) and torch.allclose(hosts[0, :, 0], torch.arange(N_HOSTS).float())
    probs = torch.zeros(1, N_ENTITIES, 2, 4, 4)
    probs[0, ENTITY_INDEX[11], :, :2] = 0.9                           # left caudate -> basal ganglia left
    probs[0, ENTITY_INDEX[12], :, 2:] = 0.4                           # left putamen, same host
    hm = R.host_masks_from_entities(probs)
    bg = HOST_NAMES.index("basal_ganglia_left")
    assert hm.shape == (1, N_HOSTS, 2, 4, 4) and hm[0, bg, 0, 0, 0] == 0.9 and hm[0, bg, 0, 3, 0] == 0.4 and hm[0].sum() == hm[0, bg].sum()


def test_geometry_between_events_and_hosts():
    # an event fully inside host 0 (left white matter), host 1 (right white matter) on the other side
    D, Hh, W = 2, 4, 8
    axes = [torch.arange(n, dtype=torch.float32) + 0.5 for n in (D, Hh, W)]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0) * 2.0           # 2 mm voxels
    host = torch.zeros(N_HOSTS, D, Hh, W)
    host[0, :, :, :4] = 1.0
    host[1, :, :, 4:] = 1.0
    ev = torch.zeros(2, D, Hh, W)
    ev[0, :, 1:3, 1:3] = 1.0                                           # inside host 0
    ev[1, :, :, 3:5] = 1.0                                             # half in each
    g = R.geometry(ev, host, coords)
    assert g.shape == (2, N_HOSTS, R.GEO_DIM) and R.GEO_DIM == 15
    assert g[0, 0, 0] == 1.0 and g[0, 1, 0] == 0.0 and g[1, 0, 0] == 0.5 and g[1, 1, 0] == 0.5
    assert g[0, 0, 4] == 0.0 and abs(g[0, 1, 4] * 100 - 27 ** 0.5) < 1e-3       # centroid (2, 4, 4) mm to voxel (3, 3, 9) mm
    assert g[0, 0, 5] == 1.0 and g[0, 2, 5] == 0.0 and g[0, 2, 4] == R.DIST_CAP_MM / 100
    assert g[0, 0, 6] < 0 and g[0, 1, 7] == 1.0 and g[0, 0, 7] == -1.0           # event on the left; host sides by identity
    assert g[0, HOST_NAMES.index("brainstem"), 7] == 0.0 and all(g[0, i, 7] == (-1.0 if HOST_SIDE[i] == "left" else 1.0 if HOST_SIDE[i] == "right" else 0.0) for i in range(N_HOSTS))
    flipped = R.geometry(ev, host.flip(-1), coords)                                 # the right host at small x: the event is now on the right
    assert flipped[0, 0, 6] > 0 and flipped[0, 1, 7] == 1.0
    one_side = host.clone()
    one_side[1] = 0.0                                                               # a crop without any right host
    assert (R.geometry(ev, one_side, coords)[:, :, 6] == 0.0).all()
    assert g[0, 0, 8:].sum() == 1.0 and g[0, 0, 8 + TISSUES.index("white_matter")] == 1.0
    disp = g[0, 1, 1:4] * 100
    assert disp[2] > 0 and abs(disp[0]) < 1e-4                                 # host 1 lies at larger x, same z


def test_candidate_competition_shapes():
    torch.manual_seed(0)
    cc = R.CandidateCompetition(16, layers=2, heads=2)
    logits = cc(torch.randn(3, 16), torch.randn(N_HOSTS, 16), torch.randn(16), torch.randn(3, N_HOSTS, R.GEO_DIM))
    assert logits.shape == (3, N_HOSTS + 1) and torch.isfinite(logits).all()
    assert cc(torch.randn(0, 16), torch.randn(N_HOSTS, 16), torch.randn(16), torch.randn(0, N_HOSTS, R.GEO_DIM)).shape == (0, 14)
