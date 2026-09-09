# STATUS:2026-09-09 交接点(tag `handoff/2026-09-09`)

每次交接前整体重写本文件。五段固定:已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

- **数据**。全库副本在 `/data2/congcong/data/FM_data`(5.85 TB,校验一致)。脑侧 SynthSeg-robust 伪标签五库完成:fastMRI 标注卷 996/997、PDGM 501、BMSR 461、HCP 1113、ISLES 250(`derived/synthseg/<ds>/seg_native/`,33 类 aseg 粒度,**没有脑叶**)。SKM-TEA M1 导出 155/155(`derived/skmtea/m1/`,44 GB;每卷 `image_clean_e{1,2}`、`image_noise_q{1,2,3}_e1`(k 空间加噪 0.25/0.5/1.0,只加在采集支撑上)、`image_us{4,8,16}_e1`(Poisson,adjoint SENSE)、`seg.nii.gz`、`boxes.csv`;`manifest.csv` 155 行全 ok)。
- **代码**。数据引擎 + arm-B 最小可训通路;130 测试通过(命令见 `CLAUDE.md`)。
- **fold-0 smoke run**(`runs/armb_fold0_first`,300 步,682 s):无 NaN,八项损失全降;host acc 0.652 是 `NOT_EVIDENCE`(真值几何)。
- **实测 2026-09-08**(`docs/verification/2026-09-08/REPORT.md`):不看类别的 oracle 重叠绑定器组织族级 0.860(fold 0,n=57;裁块内 0.804);14% 实例的标注宿主不是其框重叠最多的结构;M1 判据定在组织族级(§13.4);重叠答错的实例一律保留(§13.5)。
- **实测 2026-09-09**(`REVIEW_expert_comments_audit_2026-09-09.md` §2,`docs/verification/2026-09-09/`):**类别感知查表 oracle(候选按病灶类别限定,零重叠取最近结构)组织族级 0.968(n=309,全部 5 折 0.955–0.984);半月板撕裂 1.000(n=101,D5 规则②使类别蕴含组织族);软骨病变 0.952(n=208);侧别 0 错。可争空间约 3 个点;M1 的 10 点门槛在干净数据上不可达(§9.7:d=0.03–0.05 需 470–1500 例,手上 309)。** MTR_110 ann 15 被最近结构兜底答对;MTR_020 ann 67 仍答错。复现:

  ```bash
  cd /data0/congcong/code/Project_Doing/foundation_model
  PYTHONNOUSERSITE=1 ~/anaconda3/envs/nvgen/bin/python \
    docs/verification/2026-09-09/lookup_ceiling.py /tmp/lookup_ceiling.json
  ```

- **专家评论审计**:第二段对仓库的 12 条断言全部属实;七条外部引用全部核到;第一段四条已过时。
- **仓库整理**:审计分支 `review-expert-comments` 已合入 main 并删除;本文件与 `CLAUDE.md` 随本次交接提交。

## 2. 待用户拍板

- **A) M1 判据怎么改。** 推荐:门 = Gate 3(公平管线在退化下确有绑定失效,按"漏检"与"检出后绑错"分开报)+ Gate 4(E 在每一退化档内显著优于 softmax、熵、全局/局部 NRMSE、ConfidNet 类 learned failure prediction);"绑定 vs 查表"只作退化梯度上的曲线报告,不设门槛。维持现判据等于预先注定阴性。
- **B) 是否修改方案文档。** §9.1/§9.6 写入类别感知查表基线;§13.4 的"14% 空间"改为约 3%;§13.5 的例子由 MTR_110 ann 15 换成 MTR_020 ann 67。
- **C) 是否为脑侧重跑 SynthSeg `--parc`。** 与 `--robust` 可同用(README 里"不适用于 --robust"只针对 `--fast`),得到 Desikan-Killiany 皮层分区再聚合成脑叶;PDGM/BMSR/HCP 共 2075 卷,CPU 单进程约 15 h、4 进程约 4 h。只在决定继续第二战场(BMSR 多灶)时做。
- **旧遗留(多次未答)**:Q9 删除授权(SKM-TEA 2.4G truncated 残留 + 820G 原 tar,两处副本都有);fastMRI 其余 4850 卷未标注脑要不要跑 SynthSeg;多线圈运动仿真谁写(建议先出物理设计页);是否删除已合并的旧分支(plan-v2/v3/v4/v5、armb、review-cui-feasibility 都已在 main 里)。

## 3. 下一步(按顺序;A 定了才动第 2 步之后的)

1. 三处 teacher forcing 换预测量:`armb.py` 两处 `MINIMAL PATH`(几何用预测 mask 与预测框)+ 存在性门控用预测 presence。评估计入漏检(§9.5)。
2. 臂 A = nnU-Net 分割 + 3D 检测 + **类别感知查表**(类别限候选 + IoA argmax + 零重叠取最近);两臂训同一套退化视图,同参数量级,都从零训练(A4)。估 2–3 周(nnU-Net 五折约 1–2 GPU 天/折,可并行)。
3. 对照四臂同上游预测:B0 类别感知查表 / B1 pair MLP / B2 无几何关系 Transformer(`geo` 置零)/ B3 全量;另报 B0-naive(不看类别)以说明稻草人差多少。
4. Gate 3 → E:失效概率口径(冻结教师的关系是否仍正确),标签来自折外教师(交叉拟合),按退化档内评估 AUROC/AURC/Brier;NRMSE 局部保真度降为辅助回归。→ Gate 4。
5. 硬负样本、运动仿真、脑侧、S/U_Q 头,都在其后。

## 4. 坑与别重做

- **轴约定**:导出 (X,Y,Z)=(256,256,160),模型工作在 (Z,Y,X);数组、框、间距同步置换。间距逐卷从 header 读(0.6249083/0.625/0.8006518;MTR_049/066/095/173 是 0.7032),永远别硬编码 0.625。
- 强度未归一(min 2.3e4 / max 5.3e7),逐卷 [0.5, 99.5] 百分位裁剪 + z-score。
- 导出续跑判据是 `<scan>/boxes.csv` 是否存在;任何中断后跑一次"有 boxes.csv 但不在 manifest"的交叉检查。
- MTR_150 帧门槛失败已破案(16 线圈组 + 髌软骨三处全层缺损),变换正确,不改。
- fastMRI 脑左右手性未知,按放射学约定 (L,P,S) 假设;16 层只盖侧脑室到颅顶,实为 2.5D。
- SKM-TEA 原始 k-space 采集支撑只有 38.5%;X^0 是数据集 `target` 重建,退化图是 adjoint SENSE,做 E* 前统一重建口径。
- 欠采倍率跨扫描排不出损伤顺序(NRMSE us4 0.17–0.32 与 us16 0.22–0.41 重叠),卷内顺序成立;噪声档可排(q1 0.064–0.068 / q2 0.130–0.136 / q3 0.266–0.282)。U_Q 强度头拿 R 当标签会排不出真实损伤序。
- τ_E 无处定标:155 扫描 = 155 受试者,无重复扫描。
- 积液 116 / 韧带 38 例目前对 U_B 呈背景(计划文档 F5),正式跑分前要定处理。
- `RESEARCH_PLAN.md` §12 的 M0 状态过时(SynthSeg 已完成);§13.4 的 0.860 不是公平基线。
- 别信 `val_host_acc_NOT_EVIDENCE`。别把最小通路的规模(embed_dim 32、约 4M 参数)当正式配置(A5)。
- 侧别对任何重叠类查表都是白送的;标签级 ABA 没有独立信息。

## 5. 关键决定的为什么

- 关系真值用 `tissue_id` 不用重叠率:重叠定义会把真值送给查表管线,M1 失去意义。
- M1 判据取组织族级:侧别由 D5 规则用重叠解析,对重叠基线是循环的。
- 重叠答错的实例保留(§13.5):逐个剔除等于构造偏袒基线的测试集。
- v2.1 的 1A/2A/3B/4B:U_Q 独立全局分支、无 R_Q;统一本体 + 存在性门控;退化图的关系监督由外部 detach 的 E* 门控;E 输入与 E* 共用病灶局部 ROI。
- 两臂都从零训练(A4):否则臂 B 的胜利是权重或数据优势,不是关系建模优势。
- 09-09 审计的结论:`located_in` 关系在有分割可用时本质是几何查表;项目非平凡的内容在退化后预测几何失效时的绑定鲁棒性、无宿主 mask 的关系、以及关系可靠性 E,不在干净数据上的绑定差。这就是 A 项要重定判据的原因。
