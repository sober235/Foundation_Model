# AnatoBind-MRI 完整技术方案 V7

## 从“几何失败是否真实发生”到“关系模型能否救回”的脑 MRI 证据链重构版


## 0. 先给结论：V6 不能原样继续，但核心想法尚未被否定

REPORT.md 对 V6 的审查结果不是“方案整体失败”，而是把适用条件收窄了。V6 在 SKM-TEA 膝关节上的核心前提没有被激活：采集退化并没有造成足够多的几何宿主错配，因此继续堆叠 Relation Transformer 与 reliability head 缺乏科学必要性。相反，在 fastMRI+ 脑 FLAIR 小病灶上，噪声 q3 使 780 个小病灶中约 7.4% 的几何查表答案相对 clean 发生改变，且白质与皮层分割仍保持约 0.91 的 Dice，说明这里出现了“图像没有彻底报废，但几何宿主开始不稳定”的工作区间。

因此 V7 的核心修改不是换一个更复杂的网络，而是把研究对象、监督语义、实验门槛和模型结构重新对齐。主线应从“跨器官通用的 anatomy–abnormality binding”收缩为“脑小病灶在采集退化下的 lesion–anatomy binding recovery”。膝关节保留为负对照，不再作为主战场；fastMRI+ 脑作为主验证域。可靠性建模继续保留，但只有在 Relation Transformer 已经被证明能够真实救回几何错配之后才进入。

最重要的变化是：V7 不再把 clean SynthSeg 查表结果叫作 ground truth。它只能是 clean-reference pseudo target。没有放射科医生的独立宿主/位置标注时，论文能够主张的是 degradation consistency 或 binding recovery toward clean reference，而不是 clinical correctness。


## 1. REPORT.md 对 V6 的四个决定性修正

第一，fastMRI+ 标注坐标方向存在此前代码没有处理的上下翻转。官方 fastMRI+ README 明确说明，生成 DICOM 时图像数组做过 up/down flip，因此 CSV 标注直接叠加到原始 RSS 数组时必须转换。这个问题会直接污染病灶检测器训练，所以此前 leg 2 / H1 的失败归因必须重审。V7 将坐标完整性从“预处理细节”提升为 Gate 0：任何检测、关系或退化实验之前，必须先通过方向与坐标单元测试。

第二，脑小病灶确实比膝关节更容易触发几何宿主改变。位移探针显示，真实脑小病灶在约 2 mm 面内位移时，宿主答案改变率约 5.7%；3 mm 时约 8.2%。这意味着 V6 所需的“几何冲突子集”在脑中存在，而在膝上基本不存在。

第三，当前欠采样实验不能用于证明几何鲁棒性。zero-filled RSS 在 4× 时已经造成大范围鬼影，8×/16× 甚至使 SynthSeg 基本失效。此时测到的是 image collapse，而不是结构轻度位移。因此 V7 删除 zero-filled RSS 作为主实验条件；欠采样必须改成 raw multi-coil k-space + sensitivity estimation + 统一的 SENSE/ESPIRiT 类重建，再重新标定退化等级。

第四，当前只完成了 Gate A：证明 geometry 会坏；并没有证明 Relation Transformer 会救。B3 尚未测试。V7 的研发顺序因此必须非常严格：先修坐标与病灶端点，再用受控 relation-only 实验验证 Gate B；只有 Gate B 通过，才允许投入 reliability、motion、protocol shift 等后续模块。


## 2. V7 重新定义后的科学问题

V7 只回答一个可以被明确证伪的问题：

当同一患者、同一病灶的生物学位置不变，但采集退化导致 anatomy segmentation / lesion localization 的预测几何发生偏移时，能否利用 lesion 语义、anatomy 身份、局部图像证据和候选宿主之间的竞争关系，把 degraded view 的 lesion–anatomy assignment 拉回到 clean-reference assignment？

这一定义包含三个层次。第一层是 geometry failure：退化必须真实改变几何查表结果，而且图像仍处于可解释、非崩溃状态。第二层是 relation rescue：关系模型必须在 geometry-conflict cases 上把一部分错误宿主改回，并且不能大量破坏原本正确的样本。第三层才是 reliability：模型必须识别哪些 rescue 值得自动接受，哪些仍然缺乏足够证据。

由此，V7 的主要终点不再是“总体 host accuracy 提高多少”，而是更直接的 rescue / harm 分解。若 q3 约有 7.4% 的 780 个病灶发生 geometry conflict，即大约 58 个病例；即使关系模型救回其中 30%，总体准确率也只会上升约 2.2 个百分点，但它在冲突子集上的机制贡献却是明确的。只看总体平均分会把真正的科学信号稀释掉。


## 3. 数据与监督：三层证据，不再混用

V7 将监督来源明确分成三层。

Level P：fastMRI+ 原始病灶框。它们用于病灶存在与位置端点，但必须先修正 up/down flip。官方说明也提示，这些病灶标注来自单名放射科医生，不能被视为完整临床 ground truth，因此不应进一步推导出“宿主结构真值”。

Level C：clean-view pseudo relation。对 clean FLAIR 运行稳定的 anatomy segmentation，再根据预先定义的 anatomy ontology 产生 clean-reference lesion–anatomy assignment。该标签可用于大规模 relation pretraining / consistency training，但在论文中必须明确叫 pseudo target 或 clean-reference target。

Level R：radiologist relation reference。最终需要独立的人标关系验证。最理想是对主测试集全部 lesion 标注；若资源有限，则至少覆盖所有 geometry-conflict lesions，并加入 2–3 倍匹配的 non-conflict controls。标注者应只看 clean image 与病灶框，不看模型输出；建议双阅片或至少对高争议病例做 adjudication。

这三层标签的职责必须分开：P 解决 lesion endpoint；C 解决大规模训练；R 决定能否把“与 clean 一致”升级成“临床上更正确”。


## 4. Gate 0：先修坐标，再谈模型

V7 新增一个强制 Gate 0。任何训练前必须完成以下自动检查：fastMRI+ box 从 DICOM 标注坐标转换回 RSS 原生坐标；image、box、crop、resize、flip 的变换完全共享；随机抽取病例自动生成 overlay；程序单元测试同时检查 box 中心、边界和图像数组维度。

修复后必须重新跑此前 H1 检测器实验，因为原来的 detector 训练在镜像 box 上。此前“检测器塌陷主要由损失量级不平衡导致”的解释在坐标 bug 被修复之前不能保留为定论。

V7 的 relation 研究不等待 detector 完全成熟。为了把问题拆干净，Gate B 采用双轨评估：Controlled track 使用修正后的人工 lesion boxes，专门测试“给定病灶位置后，relation 能不能救 geometry”；End-to-end track 再使用 detector 预测 boxes。前者是机制上限，不代表部署性能；后者才是最终系统结果。


## 5. Anatomy ontology：把“宿主”与“地标”分开

V6 把全部 anatomy label 都放入候选宿主，其中包含 CSF / ventricle 等类别，导致 clean view 里也出现病灶被分到 CSF 的现象。V7 不建议继续把所有 33 个 SynthSeg 标签等价地当“host”。

建议使用两级 ontology。一级 primary host 只包含脑实质结构，例如 cerebral white matter、cortical gray matter、deep gray matter、brainstem、cerebellum，并保留左右侧信息。二级 fine anatomy 保留 SynthSeg 细粒度标签用于机制分析和消融。Ventricle、CSF、cortical surface 等不作为 primary host，而作为 landmark，用来提供“距脑室多远、距皮层多远、是否跨越组织边界”等关系证据。

这样做有两个好处：一是避免“病灶宿主 = CSF”这种语义上不自然的结果；二是把真正具有临床意义的空间关系显式保留下来，而不是全部压缩成一个 overlap winner。


## 6. Perception 层：不再让 relation 模块替上游擦屁股

Anatomy 端可以继续使用固定身份 query + 3D encoder 的框架，也可以在第一阶段直接使用成熟的 anatomy segmenter 生成 mask。关键不是哪个网络最漂亮，而是要有稳定、可复现的 degraded anatomy prediction。由于文献已经指出 SynthSeg 在低 SNR / 低组织对比时会失败，并存在更强调 robustness 的 SynthSeg+ 类方法，V7 必须加入“更强 anatomy segmenter”作为竞争基线。若仅仅升级 segmentation 就能消除大部分 geometry conflict，那么复杂 relation model 的必要性会被削弱，这个可能性必须正面检验。

Lesion 端仍然可以采用 DETR-style anonymous queries，但必须在方向 bug 修复后重训。对于 Gate B 的 controlled track，直接使用 corrected GT boxes 以隔离 detector 误差；对于 end-to-end track，再使用 predicted boxes。

24 卷 × 7 个退化视图不是 168 个独立患者。患者数仍然只有 24，不能把它当成足以从零训练 3D Swin + DETR + Relation Transformer 的主训练集。24 卷只用于探针与 operating point 选择。正式训练需要扩展到全部符合条件的 fastMRI+ brain FLAIR 病例，并坚持 patient-level split；若数据规模仍有限，应冻结或预训练 encoder，而不是用更多退化视图制造“伪样本量”。


## 7. Geometry representation：从 5D 粗几何升级为有方向、有不确定性的关系几何

V6 的 Gij = [Δz, Δy, Δx, d, IoA] 可以保留，但不应再作为完整几何描述。fastMRI+ FLAIR 的面内分辨率约 0.6875 mm，而层厚约 5 mm，z 方向定位误差与 x/y 完全不同。直接使用各向同性欧氏距离 d 会掩盖这种不确定性。

V7 建议 relation geometry 包含：三轴物理位移 Δx/Δy/Δz；lesion box 与 candidate anatomy 的 overlap / soft-overlap；lesion center 到 candidate surface 的 signed distance；到 cortex / ventricle 等 landmark 的距离；anatomy segmentation confidence；lesion box size 与 spacing。所有量均用物理单位或明确归一化。

但要强调：geometry 只是一类证据。V7 的科学目标正是当 geometry 开始不可靠时，使用 semantic embedding 与 local image evidence 补偿，因此不能让 geometry 设计复杂到重新变成一个更强的手工查表系统。


## 8. Relation architecture：从全局 K×M 混合改为“每个病灶内部的候选宿主竞争”

V6 将全部 K×M pair tokens 一起送入全局 Relation Transformer。这个设计可以工作，但它混入了“不同病灶之间”的共现先验，使得最终提升究竟来自 host competition 还是 lesion distribution prior 不够清楚。

V7 的主模型改为 per-lesion Host-Competition Transformer。对第 j 个 lesion，构造 K 个候选 token：
r_ij = Fuse(a_i, u_j, g_ij, f_local,ij)。
其中 a_i 是 anatomy identity / case-specific embedding，u_j 是 lesion embedding，g_ij 是上节几何，f_local,ij 是 lesion 与 candidate anatomy 交界区域的局部图像特征。然后只在该 lesion 的 K 个候选 host 之间做 self-attention，输出 K-way host distribution。

跨 lesion 的全局上下文不删除，但降级为 ablation：如果加上 inter-lesion context 后稳定提升，再纳入最终模型。这样主贡献更聚焦：关系模型首先做的是“这个病灶在多个候选宿主之间如何竞争”，而不是利用同一患者其他病灶的共现模式猜答案。

如果训练数据只有唯一 host 标签，V7 暂时删除独立 Rij + LrelBCE 主头，只保留 host softmax。因为在没有 secondary-relation annotation 时，两者高度冗余。等未来有人标多关系语义时再恢复 edge-level relation head。


## 9. Degradation engine：只保留“生物学真值不变、图像仍可用”的退化

Noise：保留当前 k-space complex Gaussian noise 路径。q1/q2/q3 已给出单调的 geometry-change 趋势，其中 q3 首次通过 Gate A。正式实验应重新在更大患者集上标定，并加入视觉质量 / 结构可辨认性检查，避免只用 NRMSE 判断“是否临床合理”。

Undersampling：删除 zero-filled RSS 主实验。新的欠采样流程必须从 raw multi-coil k-space 出发，用 ACS 估计 coil sensitivity，并对 full / undersampled 数据使用同一类重建算子，例如 CG-SENSE 或 L1-ESPIRiT。加速倍数应由“可用区间”反推，而不是预先坚持 4×/8×/16×；只保留 anatomy 不发生整体崩溃、但 geometry conflict 可测的 operating points。

Motion：继续放在后续阶段。只有 noise 与正式重建后的 undersampling 已经完成 Gate A/B，才加入 k-space segmented rigid motion。所有 motion view 必须使用相同 reconstruction pipeline。

V7 对退化的判定标准不是“越差越好”，而是存在一个窄窗口：退化足以破坏局部几何，但不足以让整个 perception 系统崩溃。


## 10. Gate A：重新定义“几何失败被激活”

Gate A 由三个条件同时组成。

A0，坐标正确：所有 box / segmentation overlay 通过方向与空间一致性检查。
A1，图像仍可用：主脑组织 segmentation 不应出现系统性 background collapse；可预注册大结构 Dice / QC 下限，并用人工视觉抽查确认。
A2，geometry conflict 足够多：相对 clean-reference 的 host change rate 达到预设效应量，并且 patient-level bootstrap 置信区间支持该变化。

noise q3 是当前唯一明确满足这一逻辑的条件。us4/us8/us16 虽然 change rate 更高，但 perception 已接近或完全崩溃，不应计为“成功触发”。


## 11. Gate B：Relation Transformer 是否真的“救回”几何错误

所有 B0–B5 方法必须共享完全相同的 lesion boxes、anatomy outputs 和数据 split，避免把 perception 改善误认为 relation 改善。

B0：degraded geometry lookup。
B1：candidate token 独立 MLP，不做候选之间交互。
B2：direct local ROI → host classifier，不显式使用 anatomy binding，用来检验复杂关系建模是否真的必要。
B3：Host-Competition Transformer，semantic + geometry。
B4：B3 + local boundary evidence。
B5：clean geometry oracle，上限参考。

主分析必须放在 geometry-conflict subset，并同时报告 rescue rate、harm rate、net rescue、host accuracy。定义：
rescue = B0 错且模型对；
harm = B0 对且模型错；
net rescue = rescue − harm。

如果模型只在整体平均分上提高，但在 conflict subset 没有明确 rescue，V7 的核心机制不成立。反过来，如果 conflict subset 明显提高而总体只增加 1–3 个点，这是合理的，因为冲突病例本来只占少数。统计检验应以 patient 为 resampling unit，而不是把同一患者的多个 lesion 当独立样本。


## 12. 一个必须加入的强竞争基线：更稳健的 segmentation 能否直接解决问题

V7 必须加入 stronger-perception baseline，这是原 V6 缺失的关键对照。

逻辑是：REPORT.md 证明了当前 SynthSeg-style anatomy prediction 在 q3 发生足以改变 host lookup 的偏移，但这并不自动说明需要 Relation Transformer。另一种更简单的解决方法是换一个在低 SNR 下更稳健的 anatomy segmenter。SynthSeg+ 等工作正是为低 SNR / poor tissue contrast 场景提高稳健性。

因此需要比较：
当前 anatomy segmenter + geometry lookup；
更强 anatomy segmenter + geometry lookup；
当前 anatomy segmenter + relation rescue。

只有当 relation rescue 在强 perception baseline 之上仍然有价值，或者能以更低计算/标注代价达到相似恢复效果，显式 relation 模型才有充分必要性。


## 13. Gate C：可靠性从 edge-level 改为 lesion-event level 优先

V6 的 Eij 对 K×M 大量负边做可靠性监督，容易被 easy negatives 主导。V7 更建议先定义 lesion-event reliability：
E_j = P(top-1 host assignment is correct | current evidence)。

输入包括 top-1 与 top-2 relation token、host margin、local ROI evidence、lesion detection confidence、anatomy confidence。它直接对应最终临床决策：这个 lesion 的 anatomical assignment 要不要自动接受。

监督仍采用 patient-level out-of-fold correctness，禁止同一个模型在训练样本上自己产生 correctness 标签。若只有 pseudo target，则 E_j 学到的是“相对 clean reference 是否正确”；在人标测试集上再评估它对真实 correctness 的迁移。

Gate C 的核心不是 AUROC 本身，而是同一 degradation bin 内的 selective behavior。必须报告 Brier、ECE、risk–coverage、AURC、selective accuracy 与 fixed-risk coverage，并在相同噪声等级内验证 E_j 能区分正确/错误 relation，而不是只学“图越糊越不可信”。


## 14. 训练路线：每一步都有停止条件

Stage 0：修 fastMRI+ box flip，建立 overlay/unit test；重跑 detector H1。若 detector 修复后仍不可用，先完成 controlled relation track，end-to-end 暂缓。

Stage 1：扩展 fastMRI+ brain FLAIR cohort；生成 clean anatomy pseudo labels；确定 primary-host ontology 与 landmark features；固定 patient-level split。

Stage 2：只用 corrected GT lesion boxes + degraded anatomy outputs，训练 B0–B4 relation models；上游 perception 冻结。若 Gate B 在 noise q2/q3 的 conflict subset 不成立，则停止 relation 主线。

Stage 3：把 corrected GT boxes 换成 detector predictions，验证 end-to-end relation rescue。若性能下降完全由 detector 主导，则优先改 detector，而不是继续加 relation 模块。

Stage 4：建立正式 undersampling reconstruction pipeline，重新寻找 non-collapse operating points；重复 Gate A/B。motion 只在此后加入。

Stage 5：加入 radiologist relation reference，并把 pseudo-consistency 结论升级到 clinical correctness；同时加入 stronger segmentation baseline。

Stage 6：只有前五步成立才训练 event-level reliability E_j，做 selective prediction 与 calibration。

Stage 7：最后才考虑 scanner/vendor/protocol shift、跨序列与跨器官泛化。


## 15. 论文实验矩阵：避免“做了很多实验但没有回答核心问题”

实验必须围绕四条因果链组织。

第一条：退化 → perception geometry 偏移。报告 anatomy Dice / QC、lesion localization error、host change rate、centroid/surface displacement。

第二条：geometry 偏移 → geometry lookup 失败。重点按 lesion-to-boundary distance、lesion size、anatomy pair 类型分层，确认冲突不是随机噪声。

第三条：relation model → rescue。报告 B0–B5，在 conflict subset 与 non-conflict subset 分别计算 rescue / harm，并用 patient-level bootstrap 或配对检验。

第四条：reliability → selective decision。只在 Gate B 通过后报告 risk–coverage 与 calibration。

另外必须有两类“反证实验”：其一是膝关节负对照，说明在 geometry 不易失败的器官里 relation 模块不应产生虚假收益；其二是 stronger segmentation baseline，说明收益不是因为故意选了一个脆弱上游模型。


## 16. 一个具体病例如何经过 V7

假设 clean FLAIR 中某个近皮层小病灶位于 cerebral white matter，clean anatomy segmentation 给出其 primary host 为 white matter。加入 q3 noise 后，cortical boundary 向内偏移约 2–3 mm，简单 geometry lookup 将宿主改成 cortex。

V7 不直接接受这个几何答案。对于该 lesion，同时构造 white matter、cortex、deep gray 等候选 relation tokens。white-matter token 的几何 overlap 可能下降，但 lesion local ROI 仍显示病灶主要位于白质信号背景，anatomy identity embedding 与 boundary evidence 也支持 white matter。Host-Competition Transformer 在候选之间比较后，把 top-1 从 cortex 拉回 white matter。这一例属于 rescue。

如果另一个病例在严重 noise 下 anatomy boundary 与 lesion appearance 都模糊，top-1 / top-2 很接近，event-level E_j 应降低，系统输出 Reject，而不是把一个低 margin 的 host 当作确定结论。


## 17. 最终停止规则：这项工作什么时候值得继续，什么时候应当停

V7 只有在以下链条同时成立时才值得继续做成完整论文：脑小病灶在至少一种非崩溃退化下稳定产生可复现的 geometry conflict；Relation model 在 conflict subset 上有正的 net rescue，并且 stronger segmentation / direct classifier 不能用更简单方式完全替代；在人标关系测试集上，pseudo-consistency 的提升能够转化为真实关系正确率提升；reliability 在同一退化等级内能够识别 relation failure。

任何一个环节失败，都应收缩主张而不是继续加模块。尤其是，如果修正 box 方向并换用更稳健 anatomy segmentation 后，geometry conflict 大幅消失，那么最合理的结论不是“Relation Transformer 还不够大”，而是这个任务本身不再需要复杂关系推理。


## 18. V7 最短执行顺序

现在最值得做的事情只有四步：第一，修 fastMRI+ box flip 并重跑 H1；第二，扩展 brain FLAIR 病例并固定 clean-reference pseudo relation；第三，在 corrected GT boxes 上完成 B0/B1/direct-classifier/Host-Competition Transformer 的 Gate B；第四，若 Gate B 明确成立，再投入 proper undersampling reconstruction、人标关系与 reliability。

这四步完成前，不建议继续实现 S、UQ、motion、跨器官或复杂 calibration。V7 的价值不在于模块更多，而在于把“为什么需要 relation model”变成一个可被直接验证、也可被直接否定的科学问题。


## 参考依据

1. REPORT.md（2026-09-15 脑侧探针）。
2. AnatoBind-MRI 完整技术方案 V6。
3. Microsoft fastMRI+ 官方 README：DICOM 转换时 pixel arrays 做了 up/down flip。
4. Billot et al., SynthSeg, Medical Image Analysis, 2023。
5. Billot et al., SynthSeg+, 2023：低 SNR / poor tissue contrast 是原 SynthSeg 的已知困难之一。
