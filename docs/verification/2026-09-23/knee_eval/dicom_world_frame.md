# 2026-09-23 世界帧入口核查：DICOM 转出的 MTR_010 经 `--frame world` 规范化后，解剖模型六类全出，软骨分数 3.895

每节：结论 → 命令 → 原始输出 → 支撑的决策。前缀 `source scripts/nnunet_env.sh; PYTHONPATH=. python`。

## 1. 素材：SKM-TEA DICOM 一卷转 NIfTI

结论：`SKM-TEA_ltr/dicoms/MTR_010.tar.gz` 含 480 张 DICOM = 图像序列 "Sag DESS Low 3132" 320 张（两个回波共用同一个 SeriesInstanceUID，EchoNumbers 1/2 各 160）+ "NOT DIAGNOSTIC: DESS T2 map" 160 张。`dicom2nifti.convert_directory` 把图像序列当 4D 拒转（`ConversionError: NOT_A_VOLUME`），按 EchoNumbers 拆成两个目录后 `dicom_series_to_nifti(..., reorient_nifti=False)` 各转成功。

```
S=/data2/congcong/data/FM_data/derived/knee_infer/dicom_MTR_010_src
tar -xzf /data2/congcong/data/FM_data/SKM-TEA_ltr/dicoms/MTR_010.tar.gz -C $S
# 按 EchoNumbers 硬链接到 $S/echo1、$S/echo2，再逐目录 dicom2nifti.dicom_series_to_nifti
```

```
echo 1: 160 dicoms -> MTR_010_echo1.nii.gz (512, 512, 160) (0.3125, 0.3125, 0.8) ('P', 'I', 'L') int16 max 4810
echo 2: 160 dicoms -> MTR_010_echo2.nii.gz (512, 512, 160) (0.3125, 0.3125, 0.8) ('P', 'I', 'L') int16 max 4059
  affine: [[0.055, 0.0, -0.788, -43.192], [-0.308, 0.0, -0.141, 119.053], [0.0, -0.312, -0.0, 21.862], [0.0, 0.0, 0.0, 1.0]]
```

DICOM 头：`ImageOrientationPatient [-0.176, 0.984, 0, 0, 0, -1]`（矢状位带约 10° 倾斜），PixelSpacing 0.3125，SpacingBetweenSlices 0.800002。

## 2. 规范化 + 解剖模型（Dataset901 fold 0，MTR_010 是 fold 0 的留出扫描）

结论：`to_export_frame` 把 (P, I, L) 转成 (I, P, R)，形状仍 512×512×160；nnU-Net 自己按头信息重采样到 0.625/0.8 mm 训练间距再回写。六类标签全部出现，软骨亮度分数 3.895（`scripts/verify_skmtea_frames.py` 的帧门线是 1.5；帧错了软骨标签会落在暗处，分数掉到 1 以下）。

```
PYTHONPATH=. python - <<'EOF'
from anatobind.infer.knee import run_nnunet
from anatobind.infer.canonical import to_export_frame
from anatobind.data_engine.seg_frames import cartilage_score
# img = to_export_frame(nib.load(".../dicom_MTR_010_src/nii/MTR_010_echo1.nii.gz")); 存到 world901_MTR_010/input/case_0000.nii.gz
# run_nnunet(901, input, out, [0], 6, False); seg = out/case.nii.gz
EOF
```

```
B canonical axcodes ('I', 'P', 'R') shape (512, 512, 160) zooms (0.3125, 0.3125, 0.8)
B world-frame: labels present [1, 2, 3, 4, 5, 6] cartilage score 3.895
label voxels {1: 42778, 2: 193832, 3: 40724, 4: 37375, 5: 37162, 6: 28387}
```

体素数换算（0.3125² × 0.8 = 0.078 mm³/体素）：股骨软骨约 15.1 mL、髌骨软骨 3.3 mL、内/外侧半月板 2.9/2.2 mL，量级与成人膝一致。

## 3. 完整入口（Dataset901 + Dataset902 fold 0）

结论：`--frame world` 跑通。解剖六类全出、软骨分数 3.895；病灶表 4 条，与同一扫描的 h5 帧入口（5 条）逐条对应：内侧半月板撕裂、髌骨软骨病变、积液、外侧半月板撕裂的框在面内坐标恰为 h5 帧的 2 倍（0.3125 对 0.625 mm）、层号相同，所在结构与侧别一致；h5 帧多出的第 5 条是 52 体素的韧带小块（分数 0.54），DICOM 重建上没出。

```
PYTHONPATH=. python scripts/infer_knee.py --image $S/nii/MTR_010_echo1.nii.gz --out /data2/congcong/data/FM_data/derived/knee_infer/MTR_010_dicom_world --frame world --folds 0 --gpu 6
```

```
lesion_id	family	score	x0	y0	z0	x1	y1	z1	n_voxels	host_label	host_name	side	host_fractions
1	Meniscal Tear	0.8713	299	279	102	316	314	121	5494	5	meniscus_medial	medial	{"5": 0.7116, "6": 0.0}
2	Cartilage Lesion	0.8255	149	99	43	211	132	96	45554	1	patellar_cartilage	single	{"1": 0.2339, "2": 0.0654, "3": 0.0, "4": 0.0}
3	Effusion	0.8057	119	110	36	207	147	68	54950		none	-	{}
4	Meniscal Tear	0.6771	293	297	39	300	311	51	655	6	meniscus_lateral	lateral	{"5": 0.0, "6": 1.0}
4 lesions -> /data2/congcong/data/FM_data/derived/knee_infer/MTR_010_dicom_world
WORLD cartilage score 3.895 anatomy labels [1, 2, 3, 4, 5, 6] lesion voxels per label {1: 45554, 2: 6149, 3: 0, 4: 54950}
```

overlay：http://localhost:8765/anatobind_knee/MTR_010_dicom_world_overlay.png
/home/congcongliu/figs/anatobind_knee/MTR_010_dicom_world_overlay.png（h5 帧对照：同目录 MTR_010_clean_fold0_overlay.png）

## 支撑的决策

`--frame world` 路径可用：spec §3.4 的输入约定成立，入口不需要自己重采样。DICOM 侧的两回波同序列是数据事实，入口文档要写明"先按回波拆分"。
