The six-case U replay supports a frozen query-preserving decoder experiment with full-volume false-positive accounting before a training intervention. The evidence is local and observational: 12 of 55 pseudo-label GT instances have a fully visible natural window where a retained query connected component (CC) has mask IoU >=0.5 but the best CC after the query maximum has IoU <0.5. Eight have a good retained-query CC somewhere but no good max-map CC in any fully visible window. Low whole-query IoU therefore cannot by itself identify an upstream training cause.

The analysis covers six reproduced validation cases, 55 GT, 83 natural sliding windows, 408 visible GT–window pairs, and 333 fully visible pairs. Every GT has at least one fully visible natural window (maximum visible_fraction=1.0). No new inference, training, GPU work, repository mutation, patient identifiers, or individual-case results are included in this public report. All findings remain **NOT_EVIDENCE**.

At each mask threshold, matched stage comparisons use the same natural window and local GT. The cache defines mask threshold 0.3 as probability >=0.3, mask threshold 0.5 as probability >0.5, and presence retention as strictly >0.5. A query mask may contain multiple disconnected CCs; CCs use 26-connectivity. “Max-map CC” is the best CC of the per-window voxelwise maximum over retained queries. Local CC comparisons precede volume-floor and component-score filtering.

| Source | GT | Any whole-query mask IoU >=.5 at .3 / .5 | Any all-query CC IoU >=.5 at .3 / .5 | Any kept-query CC IoU >=.5 at .3 / .5 | Any max-map CC IoU >=.5 at .3 / .5 | Same-window kept CC succeeds and max CC fails at .3 / .5 |
|---|---:|---:|---:|---:|---:|---:|
| BMSR | 13 | 0 / 0 | 5 / 5 | 4 / 5 | 2 / 3 | 3 / 3 |
| ISLES | 40 | 1 / 1 | 11 / 12 | 9 / 12 | 3 / 8 | 9 / 8 |
| PDGM | 2 | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 | 0 / 0 |
| Total | 55 | 3 / 3 | 18 / 19 | 15 / 19 | 7 / 13 | 12 / 11 |

The CC columns use fully visible windows; restricting the whole-query result to fully visible windows leaves its counts unchanged. “Any” columns independently choose the best candidate/window and are not a matched stage sequence. The final column explicitly matches the same window, threshold, and local GT. At .3 it covers 45/333 fully visible pairs; at .5 it covers 31/333 pairs. Across all fully visible windows, eight GT lose every good max-map CC despite some good retained-query CC at .3; at .5 this holds for six GT.

Only 3/55 GT have a good whole-query mask at either threshold, but connected-component analysis recovers substantially more local lesion signal. At .3, the median best all-query CC IoU across GT is .467 for BMSR, .382 for ISLES, and .875 for PDGM; the corresponding best fully visible whole-query mask medians are .00333, .00244, and .865. Whole-query pollution can arise from disconnected false positives in the same query. Conversely a disconnected query mask can have good total IoU while no single one of its components has good IoU. Query quality and connected-component quality are therefore separate diagnostics. The 37 GT without any good fully visible all-query CC at .3, or 36 at .5, show an additional local quality limitation; these observations do not establish its training cause.

Presence filtering has a narrower, measurable component-level effect. At .3, three GT have a good all-query CC but no good retained-query CC in any fully visible window; this comprises one BMSR GT and two ISLES GT. Eleven fully visible pairs cross the >=.5 criterion at this stage. At .5, four pairs cross locally, but all GT with a good all-query CC retain one somewhere. Whole-query-mask analysis alone reports zero good-mask losses to presence at either threshold and would miss the component-level losses. Every natural window has at least one present query. Lowering the presence threshold is not demonstrated to improve FP-controlled detection because the discarded parent query can contain substantial global false positives.

Max aggregation demonstrably creates destructive connectivity in some matched windows. The exact analysis identifies 12 GT at .3 and 11 at .5 where a good retained-query CC becomes a poor max-map CC in the same fully visible window. It also identifies counterexamples where the whole union IoU is low but its best CC remains good: disconnected unrelated components pollute the union denominator without establishing a bridge. Across sources, the two PDGM GT retain good fully visible max-map CCs and final components; the findings do not support a universal decoding failure. Whole-volume GT–component intersections were checked separately and were not treated as detection matches.

Gaussian averaging causes GT-voxel suppression, most strongly in ISLES, but this is a different quantity from instance sensitivity. The saved probability map reconstructs exactly in all six cases as the Gaussian-weighted mean of the saved per-window maxima (maximum absolute error 0). “Any-window positive” is an optimistic upper bound over overlapping windows; it is not an alternative detector.

| Source | Total GT voxels | Any-window-positive GT voxels at .3 | Final-positive GT voxels at .3 | Suppressed GT voxels | Suppressed / any-window-positive |
|---|---:|---:|---:|---:|---:|
| BMSR | 3,712 | 3,550 | 3,451 | 99 | 2.79% |
| ISLES | 61,576 | 55,015 | 50,182 | 4,833 | 8.79% |
| PDGM | 154,242 | 148,025 | 145,910 | 2,115 | 1.43% |
| Total | 219,530 | 206,590 | 199,543 | 7,047 | **3.41%** |

At .5, averaging suppresses 9,822/198,880 supported GT voxels (4.94%). No final-positive GT voxel lacks a positive contributing window at either threshold. Averaging can also increase matched-window GT probability and recall. All 55 GT retain some final-positive voxels at .3. The 3.41% figure is GT-voxel suppression relative to optimistic support, **not a 3.41% loss in instance sensitivity**. Neither it nor local CC IoU predicts the final connectivity, component score rejection, or official U metric.

The recommended next experiment is the frozen, GT-blind query-preserving decoding counterfactual in [the authoritative next-experiment contract](../NEXT_EXPERIMENT.md). Preserve the fixed checkpoint, mask/presence/physical-volume thresholds, source score thresholds, A32, unified U instances and R13+no-host. Candidates must be scored from their own original float32 query probabilities; the present cache does not contain exact scores for every query component. Do not substitute the cached whole-volume maximum/average probabilities as candidate scores. Cross-window duplicate handling, overlap ownership and R token identity must be written before implementation. Compare full-volume hits and all false positives without GT selection; neither local oracle gains nor this proposed experiment constitutes a repair already achieved. The evidence does not yet justify a training change.

The official U criterion remains one-to-one **full-volume bounding-box IoU >=0.1** with frozen source scores. A GT-selected local **mask/CC IoU >=0.5** oracle is not official sensitivity. No bbox performance or gate claim was computed here. Limitations include two selected validation cases per source, pseudo-label targets, correlated overlapping windows, 33/55 GT concentrated in one case, source differences in lesion size and morphology, local oracles before physical-volume/score filters, exact masks at only two thresholds, and missing full float32 probability volumes for every query. Representative float16 caches were not used for thresholds or oracle decisions.

Reproduction uses CPU niceness 19 and at most four numerical-library threads. `analyze_u.py` validates reproduction flags, GT counts, packed-mask counts/intersections, kept-query unions against saved maxima, exact Gaussian reconstruction, and final component–GT intersections. `analyze_query_cc.py` computes exact all/kept query-CC oracles across all 64 queries on fully visible pairs. It labels 1,226 query masks and provably prunes 7,350 when total query∩GT / GT count cannot beat the best CC already found. `make_public_summary.py` produces source-level aggregates only. The implementation provenance is code SHA `b240962d72e589e1bc13292cfbbaac3c1efdddc1`; the cache schema is defined in `WindowAudit`, `predict_volume`, and `instances_from_probability`.

```sh
nice -n 19 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 python /tmp/aur-six-u-analysis-20261017/analyze_u.py
nice -n 19 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 python /tmp/aur-six-u-analysis-20261017/analyze_query_cc.py
python /tmp/aur-six-u-analysis-20261017/make_public_summary.py
```

`public_summary.json` is the machine-readable aggregate companion. Detailed anonymous case/GT/window rows remain local analysis artifacts and are not part of this report or its public summary.
