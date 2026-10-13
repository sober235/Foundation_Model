# AnatoBind-Brain DICOM A/U/R 公共交互原型规格

**状态：** 用户目标已并入；实现计划待用户审阅。
**依赖：** 3D A/U/R 修复见 `docs/superpowers/plans/2026-10-13-anatobind-aur-three-goal-conformance.md`。
**数据根：** `/data2/congcong/data/FM_data`。
**范围假设：** 维持本项目 N1 的脑部 MRI 范围；不扩展到 CT、其他器官或多序列融合。2D 输入指单张 2D DICOM 图像；3D 输入指一个脑 MRI DICOM series 或 3D multi-frame DICOM。每个任务处理一个 sequence。

## 1. 用户目标

用户从公开 HTTPS URL 打开网页，不需登录。用户上传一张脑 MRI 2D DICOM，或一个 3D DICOM series；网页显示：

1. **A — 解剖识别：**32 个解剖结构 mask，13 个分侧 host 作为主门。
2. **U — 异常识别：**把训练数据中已标注病灶作为一个通用异常类，输出 slice/volume 内的实例 mask，不输出疾病诊断。
3. **R — 解剖绑定：**每个 U 实例显示一个 13 类分侧主宿主或 no-host，以及宿主概率。

用户可在网页切换 2D/3D 图像、显示/隐藏 A 与 U mask、点选异常查看 R 结果，并下载相同空间中的结果文件和 JSON 记录。2D 输出为二维 mask 和面积；3D 输出为三维 mask、体积和逐实例记录。

此站是研究原型。所有现有监督含 SynthSeg 或病灶伪标签，结果标为 `NOT_EVIDENCE`。未完成独立读者研究前，网页不得宣称临床准确性或诊断能力。

## 2. 当前实现边界

- 现有 AnatoBind AUR 是脑部 3D 模型，推理入口只读 NIfTI；它没有 DICOM 解码和单张 2D A/U/R 推理。
- 仓库中的 2D 检测、3D anatomy 和 Level R 读片工具是独立流水线。Level R 网页用于盲法人工标注，不能充当公开 AUR 推理网站。
- 3D 当前 G2-A、G2-U、G3 门均未通过；其修复和复评仍按依赖计划执行。
- U 的训练来源仍限规格中有标注的脑部队列（PDGM、BMSR、ISLES）；SibBMS 的 10 人 MS 标注及 fastMRI+ 框仍是外部核查数据。网页文案必须称“候选异常”，不能暗示对任意未见病变都有效。

## 3. 统一 A/U/R 输出契约

### 3.1 2D 模式

- 输入：一张有效脑 MRI DICOM 图像，含像素间距和完整图像方向信息；sequence 可由用户选择，也可由模型预测。
- A：输出 32 个结构的二维 mask；报告 32 类分项，13-host macro Dice 为评估门。
- U：输出单类异常实例 mask；由二维像素连通区域形成实例。
- R：每个二维异常实例输出 14 类概率（13 host + no-host）和一个主宿主。
- 记录单位：面积 mm²；不得把单层 mask 伪装成 3D 体积。记录保存图像平面方向和像素间距。
- 2D 模型是独立 checkpoint；不得把单张切片复制/补零后冒充 3D AUR 输入。

### 3.2 3D 模式

- 输入：一个脑 MRI DICOM series 或可解码为 3D 的 multi-frame object；多 series 上传时必须让用户选择一个 sequence。
- DICOM 解码后调用现有 3D AUR 推理接口；保留 geometry，输出映射回输入 DICOM 的同一 3D grid。
- A/U/R 的类别、阈值口径和原规格一致；每个异常记录体积 mm³、框、宿主、侧别和 14 类概率。

### 3.3 2D 模型与标签

2D 模型不复用 3D 权重作为可直接运行的模型。由锁定的 3D patient split 先分病人、再生成二维训练样本：从项目现有训练卷抽取 axial/coronal/sagittal 切片，并加入带物理方向的随机斜切重采样；影像、SynthSeg 标签和病灶标签必须共用变换。验证/测试患者不得跨到其他平面、增强或切片中。

2D 模型使用独立 2D Swin：channels `[64,128,256,512]`、depths `[2,2,6,2]`、heads `[2,4,8,16]`、patch `4×4`、window `8×8`；保留有效像素 mask、2D local coordinates 和映射到患者坐标的 3D mm coordinates。A 为 32 anatomy queries；U 为 64 event queries；R 为 per-instance 14 类分类头。2D 与 3D checkpoint 分开训练、分开评估。训练图像面内重采样到 1 mm，训练 crop `160×160 mm`，pad 到 patch/window 倍数。

2D 训练预算和学习率先在独立 synthetic/data smoke 中验证；正式 run 开始前将 seed、global batch、seen slices、学习率、warmup、验证清单 hash 写入 run config 并锁定。首轮建议沿用 Stage II/III 预算 240k/80k seen slices 和 3D 的 base LR schedule，标记 `PROPOSED`，不得看 test 后调整。

- A 真值：将当前 SynthSeg 伪标签投影到切片；32 类输出，按现有 13-host 门分别报告。
- U 真值：将同一 sequence 可见的病灶 mask 投影并取二维连通实例；不可见序列保持 `u_supervised=False`。
- R 真值：在同一平面的 lesion 与 host mask 上运行固定的 overlap / nearest-host 规则；保持 13 类分侧 host + no-host。
- 2D baseline：`Dataset909_AUR2DReference`，在同一 patient split 和二维样本清单上训练/评估 2D nnU-Net binary lesion reference。U 阈值只由 validation 选择；2D U 门 = 每来源在 reference 的同 FP/image 工作点上，灵敏度至少为 reference − 0.05。
- 2D R controlled gate = ABA(R) ≥ 同一 2D 测试实例上的 B0。所有伪标签指标均标 `NOT_EVIDENCE`。

## 4. DICOM 输入与几何

采用现有 SimpleITK 依赖作为解码层：单个对象使用 `ImageFileReader`；多文件序列用 `ImageSeriesReader` 的 Series ID 枚举和文件名序列读取。SimpleITK 官方接口分别覆盖单图读取、series 枚举/读取；SeriesReader 默认不加载 metadata，因此服务只显式读取用于解码和 sequence 选择的白名单字段。[ImageFileReader](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ImageFileReader.html) · [ImageSeriesReader](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ImageSeriesReader.html)

统一内部输入类型：

```python
Input2D(
    pixels_hw: np.ndarray,
    origin_ras_mm: tuple[float, float, float],
    row_axis_ras: tuple[float, float, float],
    col_axis_ras: tuple[float, float, float],
    spacing_hw_mm: tuple[float, float],
    sequence_type: str | None,
)
Input3D(
    pixels_zyx: np.ndarray,
    affine_ras: np.ndarray,   # 4x4
    spacing_xyz_mm: tuple[float, float, float],
    sequence_type: str | None,
)
```

- 只接受 Modality=MR 的 brain MRI；不猜测/拼接不同 Study 或不同 Series。
- 单个 DICOM object 按解码后维度路由到 2D 或 3D。multi-frame 只有恰好表示一个 3D volume 时才接受；4D time series/perfusion/diffusion volumes 暂拒绝。多文件输入按 Series UID 分组；若检测到多个 series，网页必须选择一个。
- 校验像素矩阵、方向、间距、切片位置顺序、重复切片和强度值范围；使用 DICOM Modality LUT/RescaleSlope/Intercept 后再进入模型归一化，VOI window/level 只影响网页显示。几何缺失、冲突或不支持时返回可读错误，不静默重排或 resize。
- SimpleITK/GDCM 的 LPS origin/direction 必须在单一边界转换为 RAS；`Input3D` 始终带完整 RAS affine；`Input2D` 保留两个像素轴在患者坐标系中的方向，不构造虚假的第三维。
- 只把去标识化后的 modality、spacing、orientation、shape、sequence type 和随机 job id 写入结果；不记录患者名、ID、原始文件名、UID、自由文本 tags 或像素数据。上传 intake 的 series 以随机 opaque id 表示，原始 Series UID 只在该 job 的临时服务端状态中使用。

## 5. 公共网页/API

### 5.1 API 契约

采用 FastAPI + `UploadFile`；文件以 multipart 接收，并添加 `python-multipart`。FastAPI 文档说明 `UploadFile` 使用 spooled file，适合大文件；本服务仍需给上传体积和文件数量设置硬限制。[FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/)

- `POST /api/intakes`: 接收一个 DICOM 文件、多个 DICOM 文件或一个仅含 DICOM 文件的 ZIP；返回随机 `intake_id`、`intake_token`、去标识 series 摘要和可选 mode。支持上传后临时选择 series。series 摘要只显示服务生成的序号、维度和间距，不显示自由文本 tag。
- `POST /api/jobs`: 输入 `intake_id`、`intake_token`、`mode=2d|3d`、服务生成的 opaque `series_id`（3D 时必需）、`sequence_type`（可选）；返回随机 `job_id` 和不可预测的 `job_token`。
- `GET /api/jobs/{job_id}`: 带 job token，返回 queued/running/complete/error 和阶段进度。
- `GET /api/jobs/{job_id}/result`: 带 job token，返回 A/U/R JSON 和结果资源路径。
- `GET /api/jobs/{job_id}/assets/{asset}`: 带 job token，返回 mask overlay 或下载结果。
- `DELETE /api/jobs/{job_id}`: 立即删除该 job 输入、缓存和结果。

GPU 推理使用单一有界队列和一个常驻模型 worker；不为每个 HTTP worker 复制加载一份 CUDA 模型。队列满时明确返回 429，任务超时返回结构化错误。

### 5.2 网页功能

在 `web/aur_demo/` 提供不依赖浏览器 build 工具的 HTML/CSS/JS 页面：

- 无登录访问；DICOM 文件、多文件 series 或 DICOM ZIP 上传；series/2D/3D 和 sequence 选择；提交前显示去标识提示。
- 2D 显示原图及 A/U overlay；点选 U 实例显示宿主、14 类概率、面积和 mask。
- 3D 显示逐层切片滑块及 A/U overlay；点选 U 实例显示宿主、14 类概率、体积和 mask。
- 提供上传状态、排队位置/阶段、完成/失败信息、重新开始、删除本次 job 和下载 NIfTI（3D）/PNG（2D）及 JSON。
- 页面固定展示：`Research prototype · NOT_EVIDENCE · not a diagnosis`。

## 6. 隐私、资源与公开部署

- 用户必须提交去标识化 DICOM。公开页面提醒：像素图像可能包含烧录文字；服务不会声称自动移除这类内容。
- Upload, intake, inference 和输出放在 job 专用临时目录；默认完成 30 分钟后清理，用户可提前删。进程重启时清理过期 job。禁止持久化病人 tag、影像、mask 或 job access token。
- 初始服务上限：单 job 总上传 ≤ 1 GiB、ZIP 解压后 ≤ 1 GiB、文件数 ≤ 2048、series ≤ 1、GPU 队列 ≤ 2 个等待 job；拒绝 ZIP 路径穿越、符号链接和非 DICOM 文件。超过上限时返回明确错误。根据公开负载探针可降低上限，不可未审阅放宽。
- URL 对公网开放且无需账号；job 结果必须通过随机 token 访问；每个远端 IP 最多 5 次 intake/10 分钟、1 个活跃 job。rate-limit 状态只保存在内存，不写入原始 IP 日志或磁盘。
- FastAPI 只监听 loopback。Caddy 对外终止 HTTPS 并反代到 FastAPI；Caddy 的官方文档说明反向代理配置和 hostname HTTPS 行为。[reverse_proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy) · [HTTPS quick-start](https://caddyserver.com/docs/quick-starts/https)
- `PUBLIC_HOST` 由部署环境提供；上线后将实际 `https://PUBLIC_HOST` 写入 verification record。验收须从不在项目内网的浏览器打开 URL。
- 不把 `/data2/congcong/data/FM_data` 或训练数据目录挂到网页容器；只挂载已登记的模型 checkpoint。

## 7. 验收与证据边界

1. DICOM unit tests：一个 2D object、一个多文件 3D series、multi-frame 3D、混合两个 Series UID、重复/乱序切片、错误 modality、损坏文件、方向/spacing 回映。
2. Model tests：2D/3D 输出形状、32 anatomy labels、U instance ids、R 14 类概率归一化、输入/输出 affine 一致；2D 与 3D 用独立 checkpoint 和独立阈值。
3. Offline metrics：3D 按 2026-10-08 原 G2-A/G2-U/G3；2D 按本规格 3.3 节定义的 independent baseline gates；所有阈值在 validation 固定，test 每臂只评一次，患者不跨 split。
4. API/UI E2E：在本机上传 2D DICOM 得到 2D A/U/R；上传 3D series 得到 slice viewer、3D masks、逐实例 R；不支持输入给稳定错误。
5. Privacy E2E：响应、日志和结果 metadata 不出现原始 PatientName/PatientID/UID/文件名；DELETE 和 TTL 后临时文件不存在；猜测 job id 不能读其他 job。
6. Public URL：从外网浏览器无需登录打开 HTTPS URL；提交两个输入模式之一并得到可交互结果；HTTPS 证书有效；上传超限/队列满均被拒绝。
7. Completion requires both: public URL E2E passes and applicable A/U/R model gates pass. If model gates fail, preserve the URL only in preview mode with inference uploads disabled; report unmet goal and do not label the end-to-end target complete. R clinical claims remain blocked on independent human-reader validation.

## 8. 非目标

- 不处理 CT、X-ray、全身 MRI、多器官或跨 sequence fusion。
- 不生成疾病 diagnosis、treatment advice、“侵及/邻近”关系或无监督 open-set 保证。
- 不把 public URL reachability、UI smoke 或伪标签门结果写成临床验证。
- 不保留上传 DICOM 或病例级结果供后续训练。

## 9. 依据

- `docs/superpowers/specs/2026-10-08-anatobind-brain-aur-design.md`：N1–N15、G2/G3、现有 3D A/U/R。
- `docs/superpowers/plans/2026-10-13-anatobind-aur-three-goal-conformance.md`：当前 3D model-fix 子计划。
- SimpleITK 与 FastAPI/Caddy 官方文档见本规格第 4–6 节。
