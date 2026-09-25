# Gate 0.5 brain frame (fastMRI+ FLAIR small lesions on clean SynthSeg)

lesions 1297 (ok 1297, {'ok': 1297}), patients 165, per patient {'min': 1, 'median': 3.0, 'max': 87, 'm_eff': 29.451811873554355}
single-slice share 0.793

share of lesions with d_interface <= t (host classes of v2.6 §3, sides merged, ventricles/CSF landmarks):
  t=0: 0.420  t=1: 0.500  t=2: 0.582  t=3: 0.697  t=4: 0.749  t=5: 0.887

d_interface quantiles {'n': 1297, 'p10': 0.0, 'median': 1.2158330826193837, 'p90': 5.185617128172885}
delta_d quantiles {'n': 1297, 'p10': 0.0, 'median': 0.9722718241315028, 'p90': 5.130384067108588}

nearest host class: {'white_matter': 1147, 'cortex': 150}
lookup host (all 32 non-background aseg labels): {41: 492, 2: 497, 42: 178, 3: 97, 24: 31, 11: 1, 4: 1}
lookup host (parenchyma only, 25 labels): {41: 492, 2: 497, 42: 200, 3: 106, 11: 2}

strata:
  stratum_series=200_201: {'n': 1077, 'share_le_2': 0.5914577530176416, 'share_le_3': 0.7084493964716806, 'share_le_4': 0.755803156917363, 'share_le_5': 0.89322191272052}
  stratum_series=other: {'n': 220, 'share_le_2': 0.5363636363636364, 'share_le_3': 0.6409090909090909, 'share_le_4': 0.7181818181818181, 'share_le_5': 0.8590909090909091}
  stratum_geometry=inplane_0.62_slice_3: {'n': 32, 'share_le_2': 0.71875, 'share_le_3': 0.875, 'share_le_4': 0.90625, 'share_le_5': 0.90625}
  stratum_geometry=inplane_0.69_slice_5: {'n': 1175, 'share_le_2': 0.5787234042553191, 'share_le_3': 0.6970212765957446, 'share_le_4': 0.745531914893617, 'share_le_5': 0.8893617021276595}
  stratum_geometry=inplane_0.86_slice_3: {'n': 11, 'share_le_2': 0.5454545454545454, 'share_le_3': 0.6363636363636364, 'share_le_4': 0.8181818181818182, 'share_le_5': 0.8181818181818182}
  stratum_geometry=inplane_0.86_slice_5: {'n': 79, 'share_le_2': 0.5822784810126582, 'share_le_3': 0.6329113924050633, 'share_le_4': 0.7341772151898734, 'share_le_5': 0.8607594936708861}

t_frozen = 2 (smallest t in (2, 3, 4, 5) with share >= 0.15); hard share 0.582112567463377; majority_flag True
GATE05: DECIDE
