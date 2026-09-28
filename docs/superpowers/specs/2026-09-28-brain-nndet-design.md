# 脑侧小病灶检测器第二臂：nnDetection · 设计规格

> 状态：2026-09-28 与用户逐段讨论后定稿（四段设计各自点头，决定见 §1）。它记录**已经定下的设计**，实施计划另写。
> 上游：S2 规格 `docs/superpowers/specs/2026-09-28-brain-detector-design.md`（D1 门；D3 把 nnDetection 定为门不过时的第二臂；D9 门不过交用户）；S2 2d 结论 `docs/verification/2026-09-28/brain_detector/2d/REPORT.md`（门不过：0.3662 @ 1.636 FP/卷）；膝侧先例 `docs/verification/2026-09-23/knee_eval/VERDICT.md` §4、`docs/verification/2026-09-24/REPORT.md` §3（两处都推荐 nnDetection）。

## 0. 一句话

S2 的 nnU-Net 2d 没过 D1 门，用户选 nnDetection（Retina U-Net）作第二臂。本轮在同一批 253 卷、同一套五折上，把 1297 个注册病灶填成"一个病灶一个实例"，只训 fold 0，用默认后处理参数出框，和 nnU-Net 2d 在同一折上比：命中数至少多 14 个（0.05）才补齐其余四折；最终是否过门仍按五折 D1 判。它负责三目标里"找到异常、在哪"这半句，并给 S5 提供推理入口；解剖识别（目标 1）不归它管。不需要医生标签。

## 1. 决定记录

| # | 决定 | 一句话理由 |
|---|---|---|
| N1 | 第二臂 = nnDetection（Retina U-Net），先做脑侧；膝侧（S6）以后可复用这套工具（用户 09-28 "按照你的倾向来"） | 专为 3D 小目标检测设计，直接学"一个病灶一个框"，不靠连通域拆分 |
| N2 | 训练预算 = 先只跑 fold 0，官方默认配置，同时实测速度（用户选 A） | 最早知道这条路值不值；五折全跑要两天以上 |
| N3 | 规则 A：fold 0 验证集上、每卷假阳 ≤ 2 的工作点，nnDetection 的命中数 ≥ nnU-Net 2d 同折命中数 + 14（探针值 92/280 = 0.3286，门槛 106/280 = 0.3786）才补齐四折；最终过门仍按五折 D1（用户选 A） | 单折只有 51 卷，直接卡 0.5 会误杀或误放；和同一折的基线配对比较才看得出有没有用。0.05 是判断线，不是显著性检验 |
| N4 | 方案 1：独立 conda 环境 `nndet`（python 3.8、torch 1.11.0 + cu113、conda 装的 CUDA 11.3.1 编译器），上游主线 97a58f3 原样使用、不改源码；不用 Docker，不用 nextrelease 分支（用户 09-28 "按照你的推荐来"） | 主线 README 要求 PyTorch 1.X；nextrelease 未发布、与主线差 904 个文件；Docker 可用但写进 /data2 的文件默认归 root、nice 与线程上限要另设 |
| N5 | 只分一类（small_lesion）；病灶类型（腔隙性梗死 57 vs 非特异性白质病变 1240）以后在检出的病灶上单独做分类器（用户 09-28 "approve your suggestion"） | 与 nnU-Net 对照口径一致；腔隙性梗死样本太少；NMS 按类别分开做，两类会在同一病灶上出两个框，多出的那个在门里算假阳 |
| N6 | 实例填充 = 按病灶体素数从大到小填，小的最后填 | 探针：1297 个一个不丢，只有病灶 815 的框被覆盖后变了 |
| N7 | 训练带 `--sweep` 以拿到验证集预测；门只用我们从 `sweep_predictions` 里按 nnDetection 默认后处理参数提取的结果；sweep 调参版只作 NOT_GATE；用 model_last，不用 model_best | 主线只在 `--sweep` 时出验证集预测；在验证折上调参再在上面评估等于选择；model_best 按验证折挑选 |
| N8 | 坐标换算 = 预处理空间里框的起点加 1 格（撤销 nnDetection 的 `[min−1, max+1]` 外扩）→ 它自己的 `restore_detection` → 换成 (列, 行, 层)；合成往返与"真值当预测"两道检查不过就不开训 | 不撤销外扩，单层病灶会变成两层，小病灶的 IoU 被系统性压到门线附近 |
| N9 | 计时止损：跑几百个 batch 后推算总时长（60 个 epoch + sweep），超过 24 小时报告并问用户；等回复期间继续跑，用户说停就停，之后可从 model_last 续训 | 用户规矩：预计超过一天的任务先问 |
| N10 | 两个环境不互相导入，只用 JSON 交接；`splits_final.pkl` 只存 Python 列表 | 仓库在 nvgen（python 3.11、numpy 2.4），nnDetection 在 python 3.8（numpy 1.x），numpy 2 写的 pickle 在 numpy 1 里读不了 |
| N11 | 给 S5 的推理与门走同一条链（model_last、8 种镜像测试时增强、默认后处理参数、同一换算）；推理时在内存里把 `inference_plan` 置空以退回默认参数，不改它的任何文件；输出与 S2 `lesions.json` 同字段，逐层框 = 3D 框在每层重复（长方体），四舍五入取整，全部框按分数排序，阈值由 S5 按报告的工作点取 | 推理和评估口径一致；不覆盖已有文件是用户规矩 |
| N12 | 本轮到 fold 0 的规则 A 结论为止（过与不过都是有效结果）；五折是下一轮，代码不变，开跑前先问时长 | 会话边界清楚；五折墙钟约一天以上 |
| N13 | 顺序：先收尾 S2 并合回 main，再从 main 开 `build/brain-nndet`；规格与计划都批准后才装环境（计划第一个任务） | brainstorming 的闸门；新分支要用 S2 的代码 |
| N14 | 算力：一张卡，开跑时从空卡里挑，不抢别人的卡；nice 19，增强 8 进程，连同其他任务不超过 48 线程 | 用户 09-28 已授权 S2 用最多 4 张 80 GB 卡；fold 0 只要一张 |

## 2. 已核实的事实（探针，2026-09-28）

源码事实出自 nnDetection 只读克隆 `~/src/nnDetection`（主线 97a58f3，2025-10-27）。数据数字出自一次性探针脚本，实施时由正式代码重算并记录，重算与此处不符时以重算为准并写明。

### 2.1 nnDetection 主线

- 依赖：README 要求 python 3.8+、"Please use PyTorch 1.X version for now and not 2.0"，安装示例为 `conda install cuda -c nvidia/label/cuda-11.3.1`、`pytorch==1.11.0 cudatoolkit=11.3`、`gxx_linux-64==9.3.0`；requirements 含 `pytorch_lightning>=1.3.1,<=1.4.2`、`nnunet==1.7.1`、`SimpleITK<2.1.0`、`torchmetrics>=0.7.0,<=0.7.3`。torch 2.x 的支持（`torch>=1.7.0,<2.9`，提交 ec9155e7）只在 nextrelease 分支：`git merge-base --is-ancestor ec9155e7 origin/main` 为否。
- 只支持 3D，没有 2D 配置（README："2D data sets ... are not supported"）。
- 默认训练 `conf/train/v001.yaml`：`max_num_epochs: 50` + `swa_epochs: 10`，`num_train_batches_per_epoch: 2500`，`precision: 16`。README：高端配置每折约 1 天。
- `scripts/train.py`：只有 `do_sweep` 为真时才预测验证集。sweep 先用默认设置（`sweep_ckpt` 默认 "last"，测试时增强 `None` → 3D 取 8 种）预测到 `sweep_predictions`，再由 `BoxSweeper` 以 `mAP_IoU_0.10_0.50_0.05_MaxDet_100` 调参，写 `plan_inference.pkl`、`val_predictions`（恢复到原图）与 `val_predictions_preprocessed`。
- 默认后处理（`BoxEnsemblerSelective.get_default_parameters`）：model_iou 0.1、`batched_weighted_nms_model`、model_score_thresh 0.0、model_topk 1000、model_detections_per_image 100、ensemble_iou 0.5、`batched_wbc_ensemble`、ensemble_topk 1000、remove_small_boxes 0.01、ensemble_score_thresh 0.0。`BoxEnsembler.from_case` 先取默认值再用 plan 的 `inference_plan` 覆盖；`get_predictor` 用 `plan.get("inference_plan", {})`，为空即默认。
- `scripts/predict.py` 必须读 `plan_inference.pkl`（只由 sweep 或 consolidate 生成）。
- 框：`instances_to_boxes_np` 给出 `(d0 起, d1 起, d0 止, d1 止, d2 起, d2 止)`，起 = min−1，止 = max+1。`restore_detection`：按 `transpose_backward` 换轴，乘 重采样间距/原始间距，加裁剪偏移。
- 读图用 SimpleITK（`nndet/io/load.py`），数组轴 = (层, 行, 列)。
- NMS 按类别分开做（`batched_nms` 按类别加偏移）。
- 重采样后消失的实例在 `compute_candidates` 里被静默丢掉（只取 `np.unique(seg)` 里还在的实例）。
- 折：`<预处理目录>/splits_final.pkl`，列表，每项 `{"train", "val"}`；文件不存在时自动按 KFold（种子 12345）新建。
- 无实例的卷：预处理断言允许空；训练时只取随机背景块。
- 本机：A800/A100 80 GB（架构 8.0）；经代理 `http://127.0.0.1:7897` 可下载 torch 1.11.0+cu113 cp38 轮子、conda `nvidia/label/cuda-11.3.1`、PyPI（HTTP 200）；Docker 带 nvidia 运行时、用户在 docker 组（不用，见 N4）。

### 2.2 数据

- Dataset903 的 253 个 NIfTI：面内间距 0.625–0.862 mm，层厚 5 mm（243 卷）或 3 mm（10 卷），形状多为 320×320×16。nnDetection 会重采样到统一间距（按现有分布大概率是面内 0.6875 mm、层厚 5 mm，以它的规划为准）。
- 注册表标签：Nonspecific white matter lesion 1240，Lacunar infarct 57（28 名患者，五折各 13/14/7/14/9）。
- 重叠：66 个病灶的逐层框与别的病灶共用体素（17 卷，4442 体素），其中 2 个被别的病灶完全盖住；按 N6 填充后 0 个丢失，1 个框改变（病灶 815）。
- nnU-Net 2d 在 fold 0 上（51 卷 = 33 个病灶卷共 280 个病灶 + 18 个正常卷）：命中 92/280 = 0.3286 @ 阈值 0.6，1.549 FP/卷，正常卷 0.167 FP/卷。
- S2 2d 五折诊断（不作门）：1297 个病灶里 668 个被预测碰到，629 个完全没碰到；碰到但所在预测块框 IoU < 0.1 的 174 个，其中多少是粘连造成的，诊断没有拆开。五折命中 475 个，过 0.5 需要 649 个。

## 3. 环境

- conda 环境 `~/anaconda3/envs/nndet`：python 3.8（来自 conda-forge）；nvcc 11.3.1 用 `conda install --override-channels -c nvidia/label/cuda-11.3.1 cuda` 单独装；torch 1.11.0+cu113、torchvision 0.12.0+cu113、torchaudio 0.11.0 用官方 pip 安装包；`pip install -r requirements.txt`、`hydra-core --upgrade --pre`、`pytorch_model_summary`（README 所列来源）；nnDetection 源码 `~/src/nnDetection` 检出 97a58f3，用系统 gcc-10/g++-10 编译扩展（nvcc 11.3 支持的主机编译器上限是 gcc 10，系统默认 g++ 是 11.4）：`CC=gcc-10 CXX=g++-10 CUDA_HOME=$CONDA_PREFIX FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 pip install -v -e .`。下载一律走代理。nvgen 不动。
- 2026-09-29 修订：README 的原配方（conda 装 `gxx_linux-64==9.3.0` 与 conda 版 pytorch）在本机解不出。`~/.condarc` 用清华镜像且 strict 频道优先级，经典求解器在 conda-forge 全量索引上 28 分钟没有结果；加 `--override-channels -c defaults` 后又因环境里的 python/libgcc 来自 conda-forge 而不可满足。版本不变，只换了装法；三次尝试的脚本与日志在 `~/logs/nndet_install/`，记入 `docs/nndet_install.md`。
- 自检：`python -c "import torch; import nndet._C; import nndet"`；`nndet_example` 生成玩具数据，按 smoke 配置（`conf/train/smoke.yaml`：2 个 epoch + 2 个 SWA epoch，每个 20 batch）带 `--sweep` 训一折，确认：验证集预测生成、按默认参数从 `sweep_predictions` 提取成功、把 `inference_plan` 置空后预测退回默认参数、框文件可读。玩具数据与模型放单独的冒烟根 `/data2/congcong/data/FM_data/derived/nndet_smoke/{data,models}`，不进正式目录。实测 sweep 耗时的量级。
- `scripts/nndet_env.sh`：`det_data=/data2/congcong/data/FM_data/derived/nndet`、`det_models=/data2/congcong/data/FM_data/derived/nndet_models`（路径不含点号，README 警告）、`OMP_NUM_THREADS=1`、`det_num_threads=8`、`det_verbose=1`。不与 `scripts/nnunet_env.sh` 在同一 shell 里 source（nnU-Net v1 与 v2 都读 `nnUNet_preprocessed`）。
- `docs/nndet_install.md` 记录实际执行的安装命令、`pip freeze`、`nvcc -V` 与自检原始输出。

## 4. 数据转换（Task903_FastMRIBrainSmallLesion）

- 目录 `${det_data}/Task903_FastMRIBrainSmallLesion/`，已存在就拒绝重建。`dataset.json`：`dim: 3`，`labels: {"0": "small_lesion"}`，`modalities: {"0": "FLAIR"}`，其余字段按 README。
- 图像：`raw_splitted/imagesTr/<case>_0000.nii.gz` 从 Dataset903 原样复制，逐个比对 sha256。case 名 = h5 文件名去扩展名（不含点号）。
- 标签：`raw_splitted/labelsTr/<case>.nii.gz` 为实例图，`<case>.json` 为 `{"instances": {"1": 0, ...}}`。病灶的逐层成员框走与 Dataset903 相同的路径（`small_lesion_rows → merged_lesions → match_registry`），按 N6 从大到小填，实例编号按填充顺序从 1 连续编号；另存实例编号 ↔ lesion_id 对照表。标签头信息沿用图像。
- 断言：每个病灶至少留一个体素；实例图非零体素与 Dataset903 二值标签逐体素相同；正常卷全 0 且 json 无实例；框被覆盖改变的病灶列表（探针为 [815]）写入报告。
- 折：Dataset903 的 `splits_final.json` 转成 `<预处理目录>/splits_final.pkl`（Python 列表，N10）；断言 253 例、五折验证集与 S2 完全相同；开训前再核一次文件哈希，确认没被自动新建的折替换。
- 预处理后清点：逐卷比对预处理前后的实例数，丢失的病灶列入报告；它们在评估里仍算分母。

## 5. 坐标换算与两道检查

```
nnDetection 预测框(预处理空间,已按 transpose_forward 换轴):
    (lo_a, lo_b, hi_a, hi_b, lo_c, hi_c)      约定 lo = min − 1, hi = max + 1
撤销外扩:   lo_* += 1,hi_* 不变               → 半开区间 [min, max + 1)
恢复:       restore_detection(transpose_backward, original_spacing,
                              spacing_after_resampling, crop_bbox)
            → 原图数组轴 (层, 行, 列),浮点
换轴:       (x0, y0, z0, x1, y1, z1) = (列起, 行起, 层起, 列止, 行止, 层止)
```

- 撤销外扩与恢复在 nndet 环境的 `scripts/nndet_runner.py` 里做，输出 JSON；换轴与评估在 nvgen 里做（N10）。
- 检查一（合成往返，单元测试）：已知实例的合成卷，含单层 2×2 小病灶；经 `instances_to_boxes_np` → 撤销外扩 → 恢复 → 换轴，不重采样时必须与原框完全相等；有重采样时每个坐标误差不超过一个原始体素。
- 检查二（真实数据"真值当预测"，开训前的验证步骤）：把 Task903 预处理后的真值框当作预测，走同一条换算与 S2 的评估代码；除预处理中丢失的实例外，全部病灶必须以 IoU ≥ 0.1 一一匹配，假阳为 0。原始输出记入验证记录。
- 两道检查任何一道不过，不开训。

## 6. 训练与算力

```
source scripts/nndet_env.sh
nice -n 19 nndet_prep 903 -np 4 -npp 4
nice -n 19 nndet_unpack ${det_data}/Task903_FastMRIBrainSmallLesion/preprocessed/D3V001_3d/imagesTr 8
CUDA_VISIBLE_DEVICES=<开跑时 nvidia-smi 确认空闲的一张卡> setsid nice -n 19 nndet_train 903 -o exp.fold=0 --sweep
```

- 默认 v001：60 个 epoch（50 + 10 SWA）× 2500 batch，混合精度；只用默认 3D 方案 D3V001_3d，低分辨率方案（若生成）不用。
- 启动由 `scripts/nndet_train.py` 完成：挑空卡、setsid、启动日志写工作树 `logs/brain_nndet/`（不入库）；nnDetection 自己的日志留在模型目录。
- 计时止损按 N9。

## 7. 评估与报告

- 门与规则 A 的输入：`scripts/nndet_runner.py extract` 对 fold 0 的 `sweep_predictions` 按默认参数、`restore=False` 提取，经 §5 换算成 JSON。sweep 调参后的 `val_predictions_preprocessed`（预处理空间，恢复前）同样按 §5 换算一份，只作 NOT_GATE；不用已恢复的 `val_predictions`，因为撤销外扩要在预处理空间里做。
- 评估复用 `anatobind.eval.detection_metrics`（IoU 0.1、一对一匹配、阈值网格 0.05–0.95、FP_MAX 2、门 0.5），单一类别。真值 = 注册表原框（`gt_boxes`），与 S2 相同。
- fold 0 报告 `docs/verification/<日期>/brain_nndet/fold0/`：
  - 规则 A 结论：fold 0 上每卷假阳 ≤ 2 的工作点、阈值、命中数与灵敏度、正常卷每卷假阳；并排 nnU-Net 2d 同折（由正式评估代码重算）与 3d_fullres 同折（只报告）。
  - FROC 阈值表；分层灵敏度（不作门）：d_interface 四档、单层 / 多层、面内尺寸三分位、几何分层。
  - 配对对照表（不作门）：两个模型各取自己的工作点，280 个病灶分四组：都找到、只有 nnDetection、只有 nnU-Net、都没找到。
  - sweep 调参版的同一套数（NOT_GATE）。
  - 已知偏差：框改变的病灶、预处理丢失的实例、两道坐标检查的原始输出。实测耗时。
  - 每个数附命令与原始输出；写明 fold 0 的数只用于规则 A，不是 D1 门。
- 规则 A 通过：报告结果与五折预计时长，按 N12 等用户同意后下一轮补齐 fold 1–4，折外预测覆盖 253 卷，按 D1 判门（1297 个病灶，253 卷计假阳），与 nnU-Net 2d 五折 0.3662 @ 1.636 并排。规则 A 不过：停下，报告，交用户决定，不调参救门（同 S2 D9）。

## 8. 推理接口（给 S5）

- `anatobind/infer/brain_nndet.py` + `scripts/infer_brain_lesions_nndet.py`：输入一个 fastMRI FLAIR h5 与所用折；h5 → RSS NIfTI（`rss_h5_to_nifti`，与 Dataset903 同一函数）→ 用 nndet 环境的 python 调 `scripts/nndet_runner.py predict`：按训练时的规划做测试集预处理（写进本次输出目录）、model_last、8 种镜像测试时增强、内存里置空 `inference_plan`（默认参数）、§5 换算 → JSON → 仓库侧换轴并写 `lesions.json`。
- `lesions.json` 与 S2 推理入口同字段：`z0`、`z1`（含）、`score`、`boxes`（`{"<层>": [[row0, row1, col0, col1]]}`，RSS 帧，行从上数）。逐层框 = 3D 框的面内矩形在 z0..z1 每层重复；坐标四舍五入取整并裁到图像内；全部框按分数降序输出，不设阈值。
- 输出目录已存在就拒绝。冒烟：在 130 个有小血管病印象但没有框的检查里挑一个跑通，核对格式。
- 多层病灶的逐层形状因此变粗（约 80% 病灶只占一层）；以后若绑定太粗，可打开它的分割输出（do_seg）逐层收紧，本轮不做。

## 9. 代码布局

```
scripts/nndet_env.sh                    环境变量(§3)
docs/nndet_install.md                   安装记录
anatobind/nndet/__init__.py
anatobind/nndet/brain_task.py           Task903 构建:复制图像 + sha256、实例填充、类别 json、dataset.json、
                                        splits_final.pkl、实例对照表
scripts/nndet_prepare.py                建任务目录(拒绝覆盖)
scripts/nndet_runner.py                 只在 nndet 环境运行、不导入 anatobind:extract / gt / predict → JSON
anatobind/nndet/boxes.py                读 JSON,换成 (列, 行, 层) 框
anatobind/eval/brain_nndet.py           收集各卷结果、复用 detection_metrics、配对对照表
scripts/eval_brain_nndet.py             fold 0 / 五折评估与报告
scripts/nndet_train.py                  启动:挑空卡、setsid、日志
anatobind/infer/brain_nndet.py          推理接口(§8)
scripts/infer_brain_lesions_nndet.py    推理命令行
tests/test_nndet_*.py                   nvgen 运行,合成小卷,不读 /data2
tests/test_nndet_runner.py              pytest.importorskip("nndet"):nvgen 里跳过,在 nndet 环境里单独运行
```

## 10. 测试（每个函数先写测试）

- 转换：大的先填、每个病灶至少留一个体素；实例编号从 1 连续、类别全 0；非零体素与二值标签逐体素相同；正常卷为空；图像 sha256 一致；目录已存在即拒绝。
- 折：253 例、验证集与 S2 相同、pickle 字节里不含 numpy。
- runner（nndet 环境）：§5 检查一的合成往返（不重采样完全相等；重采样误差 ≤ 一个原始体素；含单层 2×2 病灶）。
- 换轴：手搭例子逐项核对 (层, 行, 列) → (列, 行, 层)。
- 评估：合成 JSON 上的灵敏度、假阳与手算相等；配对对照表四组计数与手算相等。
- 推理输出：逐层框在 z0..z1 每层重复、四舍五入、按分数排序、字段与 S2 `lesions.json` 相同；输出目录已存在即拒绝。

## 11. 分支与执行

- S2 合回 main 后从 main 开 `build/brain-nndet`，工作树 `../foundation_model-nndet`。本规格与计划提交在该分支。结束合回 main，打 tag `handoff/<日期>-brain-nndet-fold0`，不 push。
- 执行沿用 S1/S2：子代理逐任务、先写测试、逐任务评审；控制方自己重跑每个任务的测试并检查 `git log -1 --format=%B`；同一工作树只允许一个控制方。
- 提交作者用仓库本地配置，消息英文，不留 AI 痕迹。不删任何东西；需要删的只列清单交用户。

## 12. 本轮完成判据

1. 环境装好、版本记录在案；玩具数据 smoke + sweep 跑通，§3 的四项确认都有原始输出。
2. Task903 建成，§4 断言全过，折与 S2 一致，预处理后实例清点写入报告。
3. §5 两道检查通过。
4. fold 0 训练与 sweep 预测完成；fold 0 报告给出规则 A 结论。
5. 推理接口在一个真实 h5 上跑通。
6. nvgen 全量测试通过，`tests/test_nndet_runner.py` 在 nndet 环境通过；文档（本规格、计划、验证记录、CLAUDE.md 代码地图、STATUS）齐全；合回 main 打 tag。

## 13. 明确不做

两类输出与类型分类器（以后单独做）；调参救门；膝侧（S6）；do_seg 逐层收紧；Docker；nextrelease 分支；修改 nnDetection 源码；`nndet_consolidate` 跨折合并与五折集成推理（下一轮再定）。

## 14. 已知风险

- S2 的主要失败是漏检（629 个没碰到），不是框不准。nnDetection 必须多找回漏检的病灶才可能过门；五折要从 475 个命中涨到 649 个。
- 10 卷 3 mm 层厚会被重采样到约 5 mm，单层小病灶可能在训练标签里消失（§4 清点）。
- 阈值网格下限是 0.05（与 S2 相同，不改）。若工作点落在 0.05，说明网格限制了它，报告里写明。
- sweep 的耗时未知，冒烟时先测量级。
- 旧依赖栈（python 3.8、PL 1.4、SimpleITK < 2.1）安装可能要处理版本冲突；每一步命令与输出记入 `docs/nndet_install.md`。
