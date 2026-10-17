# Saved validation schema (anonymous)

Inputs are the original `c2_val_20261014_1344/{cases,aggregate}.json` files. `schema.json` gives every nested key, type count, optional-field count, array length range and floating-point nonfinite count. No case or patient values are present in the exported schema or analyses.

`cases.json` is `list[dict]`, length 405. Every row has these fields:

| Field | Type / meaning |
|---|---|
| `case`, `patient` | `str`; private identities, omitted from all exports |
| `source`, `sequence` | `str`; source names and sequence types |
| `n_windows` | `int` |
| `a_supervised`, `u_supervised`, `r_supervised` | `bool` |
| `seq_pred`, `seq_truth` | `int`, six-class sequence indices |
| `seq_probs` | `list[float]`, length 6 |
| `a` | `dict`, all 405 rows, including A-unsupervised rows |
| `u` | `dict` in 143 rows; otherwise `null` |
| `r` | `dict` in 134 rows; otherwise `null` |
| `e2e` | `list[dict]` or `null` |

`a` keys are `hosts:dict[str,float]` (13 named hosts), `host_macro:float`, `entities:dict[str,float]` (32 numeric SynthSeg label codes serialized as strings), `entity_macro:float`, and `n_hosts_in_truth:int`. Some entity Dice values are JSON `NaN` when both reference and prediction are empty. Loaders must ignore nonfinite values for the same nan-mean convention as evaluation. The analysis exports nonfinite results as JSON null and reports valid counts. The original A gate averages row host macro Dice over 394 A-supervised volumes; it does not pool voxel counts or include the other 11 volumes.

`u` keys are `gt:list[dict]` and `dets:list[dict]`. GT fields are `instance:int|null`, `box:list[int;6]`, `n_voxels:int`, `volume_mm3:float`, `stratum:str`, and optional `ignore:bool`. There are 824 GT rows: 718 nonignored, 106 ignored. `ignore` is present on those 106 ignored rows; use `g.get('ignore', False)`. Detection fields are `instance:int`, `box:list[int;6]`, `n_voxels:int`, `volume_mm3:float`, `score:float` (12,906 total detections).

`r` has one key, `instances:list[dict]`. Each of the 716 rows has:

| Field | Type / meaning |
|---|---|
| `instance` | `int`, local GT instance id; join only within its own parent case |
| `truth`, `r`, `b0` | `int`, host indices 0..13 |
| `r_probs` | `list[float]`, length 14, ordered by the host mapping below |
| `query_iou` | `float`; selected query dense mask **>0.5**, IoU against GT inside a GT-centred bind crop |
| `zero_overlap` | `bool`; selection fell back to maximum soft intersection |
| `in_crop_share` | `float`; fraction of the GT instance in bind crop |
| `volume_mm3` | `float`; complete GT instance physical volume |
| `stratum` | `str`: `<5`, `5-10`, `>=10`; equivalent spherical **diameter in mm**, not volume |
| `box` | `list[int]`, length 6, half-open `[z0,y0,x0,z1,y1,x1]` in volume array order |
| `local_dice` | `float`; 13-host macro Dice of **whole-volume hard A** within the bind crop |
| `matched_det` | `int|null`, pre-score-filter detection instance id matched by box criterion |
| `matched_score` | `float|null`, that detection's score |

Host index mapping: 0 white_matter_left; 1 white_matter_right; 2 cortex_left; 3 cortex_right; 4 thalamus_left; 5 thalamus_right; 6 basal_ganglia_left; 7 basal_ganglia_right; 8 brainstem; 9 cerebellum_left; 10 cerebellum_right; 11 other_deep_grey_left; 12 other_deep_grey_right; 13 no_host. Derive truth side from this mapping; there is no saved side column. There are no no-host truths among the 716 saved controlled instances.

`e2e` rows have `det:int`, `r:int`, `b0:int`, `score:float`, `truth:int|null`, `query_iou:float`, `zero_overlap:bool` (11,967 rows). The null truth means that a detection has no matched GT; it is not a no-host truth.

`aggregate.json` top-level keys are `evidence:str`, `n_results:int`, `sources:list[str]`, `a:dict`, `u:dict`, `r:dict`, `s:dict`, `not_implemented:list[str]`, `run:dict`. For replay comparison, original detection boxes, counts, voxel counts, volumes and scores must come from each `cases.json` row's `u.dets`, not an aggregate sweep. Nested aggregate A/R scores are reproduced by this analysis to numerical precision.

Unavailable in these files: selected query id, bind window coordinates, coarse masks consumed by R, A/U presence, candidate-host geometry, local-crop host target, lesion-inside/outside A errors, voxel masks, SynthSeg reference-quality judgment and human host truth.

The larger machine schema is retained in local analysis archives; this readable schema is the repository snapshot.
