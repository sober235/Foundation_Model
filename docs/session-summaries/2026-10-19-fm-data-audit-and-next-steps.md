# 统一2D/3D脑AUR项目进展与148训练数据审计总结

日期：2026-10-19。归档代码基线：`6265015`；前轮统一输入执行基线：`6e5af4e`。

本文件归档此前会话中已经完成的工作、实际观察及待办，补充代码基线中尚未更新的数据审计状态。**统一输入的工程闭环已建立，发现的数据标签缺陷已修复；正式全尺寸统一训练和A/U/R质量验收仍未完成。** 本次归档只新增总结文档，没有重新训练、重新跑测试或重新读取医学数组。

## 1. 用户目标与当前实现

目标是同一个模型、一个checkpoint，接受独立单张2D MRI或3D体数据，分别完整输出解剖A32、病灶实例U、13个分侧宿主加no-host的关系R。保留提供的DICOM原始字节、元数据及来源对应关系；模型使用的特征白名单与原件保留分开。当前限定脑MRI、单序列，网页/API后置。

已经实现：

- `dual_input=True`：真实depth=1的2D路径及原3D路径，共享主体，输入/上采样模块按维度适配。
- 两种模式的SSL和监督训练、严格旧权重迁移、单checkpoint及恢复；2D用面积mm²，3D用体积mm³。
- classic MR DICOM严格几何检查、源hash/帧映射与原网格推理；缺几何、歧义维度、非法对象显式拒绝。
- 三源各1个TRAIN患者的tiny随机模型pilot：Stage I/II/III各24步，共72次optimizer step，每阶段2D/3D各12次曝光。两维loss下降仅是技术验证，不是完整模型泛化结果。
- CPU恢复验证精确；CUDA恢复瞬间的完整状态精确，不承诺未来反向参数逐位一致。

既有证据见[统一输入执行报告](../verification/2026-10-19/aur_unified_input/RESULTS.md)及[对应实施计划](../superpowers/plans/2026-10-19-aur-unified-input-execution.md)。原完整模型A/U/R质量门仍FAIL，不能由新接口、pilot或数据修复直接改判。

## 2. 148上数据的实际构成

148为172.20.114.148；实际可读根为`/data2/congcong/data/FM_data`。此次只读该数据根，不接触冷备份。

文件清点得到342889个regular file paths、21438目录，表观7518492879178字节（约7.52TB）。334044个唯一inode，8845个重复硬链接路径；19169个符号链接均指向根内。别名、重复导出和压缩包不能当作新患者。

| 脑来源 | 实测规模 | 主要角色与限制 |
|---|---|---|
| PDGM | 495来源限定患者、501扫描 | T1/T1c/T2/FLAIR；A为SynthSeg伪标签；U按可见序列和指定标签值监督 |
| BMSR | 314患者、461扫描 | T1pre/T1post/FLAIR及subtraction；U仅T1post/T1c |
| ISLES | 250患者、250扫描 | DWI/ADC/FLAIR；当前AUR只用DWI/ADC，U仅DWI |
| SibBMS | 193患者、372扫描/session | 当前模板网格只作A/序列表征，不具现成U监督 |
| HCP | 1113患者、T1w/T2w | SSL TRAIN1001、validation112，已在已知Stage I数据中 |
| fastMRI脑 | 5847卷、92590个RSS平面 | FLAIR451卷/7150平面；8213框行涉及997卷，框不等于dense U或医生R |
| OASIS | 416患者、436扫描 | 436对Analyze强度/fseg同格点；fseg只有0/1/2/3，不能代替A32或健康阴性 |

fastMRI脑有5847个不同patient_id元数据值，XML FOR有4634个不同hash；不能据此声称5847名生物学独立患者。SKM-TEA、fastMRI膝及其它器官数据保留原任务角色，不直接混入脑A32/R13。

档案共182个：11个ZIP只读central directory，未提取payload；171个tar.gz未作流式内容全审。已展开目录的结论不能扩展为所有压缩容器内均无原生DICOM。

## 3. 分区、曝光与A类覆盖

| 角色 | 行数 | 来源限定患者数 |
|---|---:|---:|
| 有效监督TRAIN（排除val_patients） | 3499 | 894 |
| 开发validation | 405 | 101 |
| 历史test | 1056 | 249 |
| SSL TRAIN | 5501 | 1895 |
| SSL validation | 629 | 213 |

监督三角色身份交集为0，但尚未证明跨来源生物学去重。旧test已暴露，不能通过重新切成2D成为新独立队列。

PDGM/BMSR/ISLES/HCP无已知监督或SSL清单外的来源患者；SibBMS8人、OASIS416人仅是候选。fastMRI另5400个元数据ID未出现在旧447个outline/QC集合中，不等于从未被其它历史模型曝光。

标签数组覆盖实际TRAIN2108张独立anatomy图，另128张已暴露开发val地图单列；2236/2236成功，没有读取历史test dense标签。A可监督1068图/3420行，32类均有有效监督体素。

| 重点aseg类 | 剔除ignore后存在图数/1068 | 有效体素 |
|---|---:|---:|
| 5 | 1063 | 588264 |
| 44 | 1057 | 630313 |
| 18 | 1066 | 1981124 |
| 26 | 1065 | 740213 |
| 58 | 1064 | 764405 |

因此，历史五类预测缺失不能解释为全TRAIN没有这些标签。稀少体素、采样、类别竞争与优化仍需区别检验。PDGM 54个TRAIN T1c没有该序列选择的目标值，也不能因此称这些患者健康。

## 4. 已完成的ISLES标签修复

180个TRAIN ISLES的原生标签有177张阳性、3张全零，旧派生只有142张阳性。35张原生阳性因为NIfTI proxy scaling使标签略小于1，在直接转int16时被截成全零。最大整数偏差仅2.3283064365386963e-10；原目标grid和原代码能逐体素复现180/180，根因不是affine错位。

代码提交`6265015`增加[discrete_labels](../../anatobind/aur/labels.py)，并覆盖重采样、dataset、外部U评价及fastMRI参考入口：

- 浮点值先验证有限、距整数绝对偏差≤1e-5，再round；整数输入保留精度。
- 使用Python整数边界检查，拒绝概率图、NaN/Inf及输出溢出。
- 先解码类别再nearest-neighbour重采样；连续影像插值不变。
- 错误不回显病例数据；旧源文件、清单、缓存及历史hash合同不覆盖。

私有新版本位于`~/outputs/aur_data_audit_20261019/labels/repair_train_v1/`：

- 35张新mask恢复770019个体素、459个≥10mm³组件。
- 新TRAIN manifest仍3499行，更新70条DWI/ADC病灶引用。
- 3张原生空DWI仅在新清单标为U未知/R不可用，U-supervised由1216降为1213，未制造健康阴性或no-host答案。
- 新manifest SHA256为`40feb9b06a56657935a62873a7e52bb57446935b6b107e2a31f7f24fccd24f28`。
- 生产loader逐例验证实例0→459、A-ignore0→770019；entity、affine/spacing及A开关未改。恢复lesion也会修复A-ignore，不能称A训练目标完全不变。
- 三源各1个冻结TRAIN样本共6次2D/3D crop/collate通过，二维默认R资格关闭。158个输入文件hash不变。

最终同版本重放结果一致，未增加独立医学证据。开发val未应用本次标签修复，不能直接用于新正式选点。必须使用新manifest才能获得恢复目标；对旧全零派生round或resume旧导出均不能恢复信息。

## 5. DICOM几何与来源关联

已覆盖342889个文件前132byte、113409个DICOM header、7174份H5 metadata/XML、55044个原始/标注NIfTI header；连同已读派生共79570个NIfTI。31391个非当前引用legacy派生NIfTI未作完整读头。

- 全部DICOM为classic单帧MR，实际enhanced/SEG为0。
- raw fastMRI112609帧（脑70732、膝41877）全部缺IPP/IOP。
- 800份derived目录对象有几何，但有UID不合规、重复/不规则位置，没有合格规则栈；不能作为原生脑DICOM验收证据。
- 5640卷（脑4468、膝1172）建立H5路径、非空患者hash、shape/depth及索引候选关联；DICOM/XML FOR hash全部不匹配。
- 全部112609帧PixelSpacing对应XML reconSpace，其中76032帧不同于encodedSpace；没有可信上游校准前，不能直接认定任一方是真实患者尺度。
- H5虽有matrix/FOV和patientPosition，却无逐片位置/方向，不能补出患者affine。
- 当前1mm4960行和SSL6130行引用均有header且同格点；原source-grid保留12组双form冲突、1A/5U exact grid差异，现有行没有DICOM原帧链。

代表性内容QC仅限既有曝光TRAIN三卷九帧：初始严格线性量化3/9通过；同帧观察范围双端裁剪的事后拟合9/9。后者不证明无损、具体生成过程、患者左右手性或坐标；原严格失败保留。

## 6. 初始化、二维R与医生标注

已经验证Stage I backbone导出160 keys严格迁移；step8500完整SSL recovery的176 keys也可严格载入，骨干与best export逐位相同，统一StageOne有4个新增二维tensor待训练。这确立warm-start资格，**不是新正式初始化run或exact resume**；旧recovery缺完整CUDA/Python/NumPy RNG。

真实二维R显式资格仍为0。现有A/U足够产生待验证的几何弱正标签，但需要按source instance、plane、FOV及变换后观测重新计算资格。宿主不可见、并列、无可靠重叠时关闭监督，不能把plane外宿主改成no-host。AUR的host_index=0有效，NO_HOST=13；几何伪参照只作训练/调试，不是医生truth。

Level R实际数据库已经存在：1297 lesions、165登记患者、2594 orders、2读者账号；labels、adjudications、releases均为0。不能再称“尚未建库”，也不能称已有人标。屏幕左右不能未经可信映射转成患者左右；本轮没有联系读者或发布自动标签。

## 7. 已报告的验证结果

本次归档沿用会话中实际执行的结果，没有重跑：

- 先7条回归RED复现标签截断/非法输入未拒绝，修复后相关检查通过。
- 独立审阅发现1项dtype边界P2，再6条RED后修复；45项针对检查通过（12.19秒）。
- 增量代码审阅0项剩余，另57项独立合成边界检查通过。
- 完整CPU：**1528 passed、5 skipped，202.88秒**。5项分别为nndet环境模块、1条CUDA bf16、3条CUDA恢复用例；本轮没有把历史专用环境补测记为新结果。
- 沙箱内首次全套因DataLoader多进程等待停滞，在377 passed/2 skipped后中断；沙箱外完整重跑通过。
- 几何独立审阅修正3项汇总/表述问题，增量复核0剩余；数据审计没有新GPU训练或模型性能结果。

## 8. 下一步建议与未完成项

建议主线为“修复数据版本 → 已验证Stage I权重warm-start → 全尺寸双模式训练”，DICOM来源与医生R并行补齐。扩大脑数据后从头联合SSL、先完整打通原生DICOM再训练是备选路线，均需额外准备。

| 缺口 | 当前变化 | 下一验收 |
|---|---|---|
| 训练数据可靠性 | 35例标签丢失已修复，TRAIN新版本与loader通过 | 开发val按冻结政策QC；防止使用旧坏派生 |
| 完整模型初始化 | 160骨干/176完整SSL keys已验证 | 新二维tensor、optimizer重置、初始化来源和种子入run合同 |
| 正式训练 | 仍只有tiny技术pilot | 正式runner、mode/source曝光、预算、恢复、双mode G1 |
| 真实二维R | 有可计算弱资格设计，生成器未实施 | 观测级资格、同观察范围基线、医生参照分开验收 |
| 原生DICOM | 全量暴露几何/尺度/UID缺口并建立候选关联 | 可信上游逐片位置/方向、尺度和原件链 |
| 医生R与独立评价 | 找到待标库，标签仍0；未建立新独立队列 | 实际读者标注、发布/封存与患者关联/曝光审计 |
| A/U/R质量 | 原完整模型FAIL保留 | 修复后新模型的两mode×三任务独立验证 |

不将旧test重新包装为新队列，不据标签恢复推断模型已经改善，不用自动几何答案冒充医生标注。

## 9. 证据保留与隐私边界

详细私有证据保存在`~/outputs/aur_data_audit_20261019/`的inventory、labels、readiness、geometry及code_review目录，医学数组、逐患者记录、UID、修复mask和manifest不上传GitHub。目录/文件按0700/0600控制。

审计中一次旧adapter源码读取曾输出9个硬编码排除病例标识，已记录为输出边界偏差，不重印，不声称整个会话匿名。本归档只包含匿名计数、规则、公共代码链接及数据版本hash。

本次独立总结分支不合并主分支，也不收取主工作区尚未提交的设计/审计文件；后续实施计划及完整报告由主工作线完成审阅后另行交付。

