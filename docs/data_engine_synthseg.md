# 脑侧解剖伪标签管线(SynthSeg-robust 2.0)

M0 阻塞项的落地记录,2026-09-05。全库脑侧细分解剖标签为零,这条管线用 SynthSeg-robust 2.0 为 fastMRI 脑、PDGM、BMSR、HCP 生成 FreeSurfer aseg 口径的解剖标签(33 类,§4.3 的 SynthSeg 粒度)。

## 环境

- `~/anaconda3/envs/synthseg`:Python 3.8.20,tensorflow 2.2.0(统一包,`tensorflow-gpu` 在 PyPI 已无 2.2),keras 2.3.1,numpy 1.23.5,h5py 2.10,nibabel 5.0.1;cudatoolkit 10.1 + cudnn 7.6.5 已装但**不用 GPU**:在 A800(sm_80)上 TF 2.2 要对 CUDA 10.1 的 PTX 做即时编译,17 分钟未完成且独占 76 GB 显存,已放弃。全部跑 CPU。
- 坑:激活 conda 环境后直接调 `pip` 会落到 `/usr/bin/pip`(系统 Python 3.10,TF 只有 ≥2.8),须用环境的绝对路径 `python -m pip`。PyPI 索引走代理 `127.0.0.1:7897`,`files.pythonhosted.org` 直连更快(设 `no_proxy`)。
- 代码:`~/src/SynthSeg`(BBillot/SynthSeg 克隆);权重见 `~/src/SynthSeg/models/PROVENANCE.md`:`synthseg_1.0.h5` 随仓库;`synthseg_robust_2.0.h5` 来自 HuggingFace 镜像 `SaiChandKasoju/SynthSeg_Robust`(官方 UCL SharePoint 链接已 404),sha256 `7bc30bf5…cb01c`。功能核验:与官方 1.0 在 PDGM/HCP 1 mm 数据上各结构 Dice 0.83–0.95;在 fastMRI 5 mm 厚层上 1.0 丢掉全部脑室与深部核团,robust 正常——所以只用 robust。
- 数据引擎代码跑在 `~/anaconda3/envs/nvgen`(torch 2.5.1 / MONAI 1.5 / nibabel / h5py / pytest 9.1),一律 `PYTHONNOUSERSITE=1`。

## 几何约定(fastMRI h5 → NIfTI,`anatobind/data_engine/fastmri.py`)

- `reconstruction_rss` 形状 (slice, row, col);行 = 读出轴(ISMRMRD reconSpace x),列 = 相位轴(y)。面内间距 = FOV/矩阵 = 220/320 = 0.6875 mm;层间距 = reconSpace fov_z = 5 mm。官方 fastMRI 脑 DICOM 证实 SpacingBetweenSlices = SliceThickness = 5,无层间隙(encodedSpace 的 z=7.5 是编码空间数值,不是层间距)。
- 渲染核实:行向下 = 前→后;切片序号增大 = 下→上。**左右手性未知**,按放射学约定假设(图像左 = 患者右),NIfTI 轴码 (L, P, S)。这条假设影响 SynthSeg 左/右标签的命名,不影响绑定关系本身;PDGM/BMSR/HCP 有真实仿射,不受影响。待用官方 DICOM 与 h5 做一次内容匹配来定。
- fastMRI 脑 16 层 × 5 mm ≈ 80 mm,实测两卷都是从侧脑室水平到颅顶:小脑、脑干、颞叶多数卷不在视野内,A 解码器的 no-object 是必需的。

## 运行

```
# fastMRI+ 有标注的 997 卷脑(全部有 raw)
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/run_synthseg_fastmri_brain.py --annotated-only --workers 4 --threads 12
# NIfTI 数据集
... --glob '<pattern>' --work-root /data2/congcong/data/FM_data/derived/synthseg/<name> [--stem-prefix-parent 2]
```

- 输出:`/data2/congcong/data/FM_data/derived/synthseg/<dataset>/seg_native/<stem>_seg.nii.gz`(原生网格,int16),`chunks/NNNNN/seg_1mm/`(SynthSeg 1 mm 输出),`chunks/NNNNN/volumes.csv`,`manifest.csv`(每卷一行,status ok/missing/error)。断点续跑:已存在 seg_native 的卷自动跳过。
- 速度:CPU 12 线程约 22–26 s/卷(文件夹模式,含分摊的模型加载);997 卷 4 worker 约 1.5–2 h;PDGM 501 + BMSR 461 + HCP 1113 用 2 worker 约 9 h。
- 2026-09-05 23:20 启动:fastMRI 标注卷(日志 `derived/synthseg/fastmri_brain/run_annotated_*.log`)与 PDGM→BMSR→HCP 链(日志 `derived/synthseg/run_nii_chain_*.log`)。
- 选用的输入:PDGM `*_T1.nii.gz`(各序列同空间,一套标签通用);BMSR `*_T1pre.nii.gz`(0.86×0.86×1.5 mm,seg 同网格);HCP `T1w_acpc_dc_restore_brain.nii`(0.7 mm,输出回采到 0.7 mm)。ISLES:FLAIR(0.71 mm,RAS)与 DWI/ADC + 病灶 mask(2 mm,LAS)不在同一网格;实测 SynthSeg-robust **直接在 2 mm DWI 上**出全 32 类且解剖合理(case 0001 抽检图 `~/figs/anatobind/synthseg_robust_isles0001_dwi.png`),故按 `*_dwi.nii.gz` 跑,标签直接落在 mask 网格上;已排在 PDGM→BMSR→HCP 链之后自动启动(日志 `derived/synthseg/run_isles_*.log`)。

## 标签本体(SynthSeg 2.0,33 类)

0 背景;2/41 大脑白质;3/42 大脑皮层;4/43 侧脑室;5/44 下侧脑室;7/46 小脑白质;8/47 小脑皮层;10/49 丘脑;11/50 尾状核;12/51 壳核;13/52 苍白球;17/53 海马;18/54 杏仁核;26/58 伏隔核;28/60 腹侧间脑;14 第三脑室;15 第四脑室;16 脑干;24 CSF。左/右 = 2–28 / 41–60。合并到方案 §4.3 的 K≈30 实体:每侧 13 + 中线 4。

## 测试

`PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q`(24 个,合成小体数据,0.6 s)。所有函数按 TDD 先写测试。
