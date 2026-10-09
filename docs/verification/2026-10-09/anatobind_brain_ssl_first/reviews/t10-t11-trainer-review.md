# Independent review of the Stage II / III trainer batch (T10, T11) — 2026-10-09

Reviewer: a fresh general-purpose agent (read-only; it ran the contract tests and its own 2-process gloo DDP probe on the CPU). Verdict: nothing blocking; plan contract and DDP safety hold. Findings and what was done:

| # | Severity | Finding | Done |
|---|---|---|---|
| 1 | should-fix | Resume discards the previous best and can leave a stale fixed-name export | best restored from `best.json` on resume; the fixed export is never overwritten, its sidecar records which file is the true best; a re-validated step with an existing export is not re-exported |
| 2 | should-fix | Resume restarts the data stream at epoch 0 while counting the budget as new | `epoch` and `epoch_step` saved in the resume meta; the resumed run skips to its position; `stop_after` added so that a run can stop without changing its schedule; a resume with a different schedule is refused (`schedule mismatch`) |
| 3 | should-fix | NaN in the JSON logs (`r_acc`, `u_matched`); R logged as sample counts, not instance counts | counts logged (`r_samples`, `r_correct`, `r_instances`, `u_matched_n`, `u_instances`), ratios only where defined, None otherwise |
| 4 | should-fix | `a_dice` counted A-unsupervised crops | gated by `a_supervised` |
| 5 | should-fix | A rank with fewer rows than one micro-batch spins forever | `make_loader` raises |
| 6 | nit | Resume does not compare the schedule | done with 2 |
| 7 | nit | `p.sum() * 0.0` gives expanded-stride gradients (DDP warning, extra copy) | `(p * 0.0).sum()` |
| 8 | nit | Weight decay shrinks parameters that only receive the zero touch in Stage II (`backbone.mask_token`, `masks.embed_coarse`, ≈ 0.69× over 15 k steps) | both frozen in Stage II (`stage_two_frozen`) |
| 9 | nit | Hungarian cost under bf16 autocast | the matching runs with autocast disabled |
| 10 | nit | Stage III warmup 200 is not in Q12 (PROPOSED); plan §2 command lacked `--g1-report`; CLI ignored conflicting flags | marked PROPOSED in code; plan commands fixed; conflicts raise |
| 11 | nit | Validation runs on every rank with the same crops | left as is (noted) |
| 12 | tests | Stage III inheritance test tautological; resume test thin; one-instance relation test; no DDP test | all added: zero-rate Stage III run equals the Stage II weights; resume reproduces the uninterrupted run's lr and losses at steps 3–4; two-lesion crop checks the matched targets; `validate` counts; sharding; a 2-rank gloo DDP test with extreme supervision in both stages |

Still open: the 4-GPU 100-step verification record with save / resume (plan T10 acceptance) waits for free cards; the same resume-epoch flaw exists in the Stage I trainer (`anatobind/aur/ssl/train.py`, not changed while the main run uses it).

The reviewer's own evidence: `tests/test_aur_training_contract.py` 11 passed (before the additions); its gloo run: every trainable parameter received a gradient on both ranks under opposite extreme batches, reduced gradients identical, no hang.
