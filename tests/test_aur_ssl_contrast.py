# tests/test_aur_ssl_contrast.py
import torch

import anatobind.aur.ssl.contrast as C


def test_projector_pools_the_valid_tokens_to_unit_vectors():
    proj = C.Projector(in_channels=8, hidden=8, out=4)
    feat = torch.randn(2, 8, 2, 3, 3)
    valid = torch.ones(2, 2, 3, 3)
    valid[1, :, :, 2] = 0.0
    z = proj(feat, valid)
    assert z.shape == (2, 4) and torch.allclose(z.norm(dim=-1), torch.ones(2))
    feat2 = feat.clone()
    feat2[1, :, :, :, 2] = 100.0                                              # an invalid token does not change the pooled vector
    assert torch.allclose(proj(feat2, valid)[1], z[1])


def test_info_nce_rewards_matching_views_and_ignores_the_patient_s_other_crops():
    torch.manual_seed(0)
    z1 = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1)
    patient = torch.tensor([0, 1, 2, 3, 4, 5])
    good, acc = C.info_nce(z1, z1.clone(), patient)
    shuffled, _ = C.info_nce(z1, z1[torch.randperm(6)], patient)
    assert good < shuffled and acc == 1.0
    z2 = torch.nn.functional.normalize(z1 + 0.1 * torch.randn(6, 16), dim=-1)
    same_patient = torch.tensor([0, 0, 1, 1, 2, 2])
    with_pairs, _ = C.info_nce(z1, z2, same_patient)
    z1_drop, z2_drop = z1[::2], z2[::2]                                        # one crop per patient: the same candidates minus the siblings
    without, _ = C.info_nce(z1_drop, z2_drop, torch.tensor([0, 1, 2]))
    all_distinct, _ = C.info_nce(z1, z2, patient)
    assert with_pairs != all_distinct                                          # the siblings are not negatives
    one_patient, acc1 = C.info_nce(z1[:2], z2[:2], torch.tensor([7, 7]))
    assert one_patient == 0.0                                                  # nothing to contrast against: zero, not nan
    loss, _ = C.contrast_loss(z1, z2, patient)
    assert torch.isclose(loss, all_distinct)
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
    l, _ = C.info_nce(z, z.roll(0, 0), torch.arange(4))
    l.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
