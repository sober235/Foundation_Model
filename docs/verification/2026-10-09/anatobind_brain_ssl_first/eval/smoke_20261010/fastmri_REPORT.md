# AnatoBind-Brain on the fastMRI FLAIR stacks (NOT_EVIDENCE: agreement with SynthSeg pseudo-labels on fastMRI FLAIR stacks (the S4 measure; report only))

1 stacks, the S4 measure (reliable slices; boxes whose slices are all reliable). The stacks are resampled to 1 mm, predicted, and the prediction is brought back by nearest neighbour.

| measure | AnatoBind | S4 student |
|---|---|---|
| mean host Dice (13 classes, reliable slices) | 0.0448 | 0.2665 |
| box host agreement (6 boxes) | 1.0000 | 0.9215 |

| class | AnatoBind | S4 student |
|---|---|---|
| white_matter_left | 0.0531 | 0.8334 |
| white_matter_right | 0.2702 | 0.8360 |
| cortex_left | 0.0286 | 0.7798 |
| cortex_right | 0.0065 | 0.7787 |
| thalamus_left | n/a | 0.0240 |
| thalamus_right | n/a | 0.0031 |
| basal_ganglia_left | 0.0000 | 0.1345 |
| basal_ganglia_right | 0.0000 | 0.0748 |
| brainstem | n/a | 0.0000 |
| cerebellum_left | 0.0000 | 0.0000 |
| cerebellum_right | 0.0000 | 0.0000 |
| other_deep_grey_left | n/a | 0.0000 |
| other_deep_grey_right | n/a | 0.0000 |

The SynthSeg reference itself barely holds the deep structures on these slices (S4 record): the Dice of thalamus, basal ganglia, brainstem and cerebellum measures the reference as much as the model.
