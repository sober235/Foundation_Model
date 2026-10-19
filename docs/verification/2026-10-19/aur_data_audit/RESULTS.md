# 148训练数据审计与ISLES标签修复

日期：2026-10-19。起点`6e5af4e`；分支`feature/aur-data-audit-and-label-repair-2026-10-19`。

## 结论

这批数据能继续支持统一2D/3D脑A/U/R开发，但不是可直接混合的统一监督库。本轮发现并修复了35个TRAIN ISLES病例的病灶标签丢失：新数据版本恢复770019体素、459个≥10mm³组件，实际生产loader逐例通过。完整Stage I权重可严格warm-start；二维R资格、全尺寸正式runner/双mode G1、原生DICOM患者几何和医生R仍须分别补齐。未启动正式训练、未产生新A/U/R性能成绩。

设计与后续边界见[数据驱动修复设计](../../../superpowers/specs/2026-10-19-aur-data-grounded-gap-closure.md)。既有tiny训练及旧完整模型结果见[上一轮执行报告](../aur_unified_input/RESULTS.md)，本轮不重复其GPU测量或改写旧FAIL。

机器可读证据见[匿名数据汇总](data_assets_anonymous.json)、[命令与hash合同](RUN.json)和[提交产物检查](public_verification.json)。

## 数据实际构成

用户给出的148即当前主机；可读根为`/data2/congcong/data/FM_data`。全目录清点342889 regular file paths、21438目录、表观7518492879178字节；334044个唯一inode、8845个重复硬链接路径。19169个符号链接均在根内，17997个文件别名、1172个目录别名；均不作为新增患者。名称/器官启发式只作初筛，fastMRI实际按H5 acquisition重分脑/膝。

| 脑来源 | 原始规模 | 序列与监督角色 |
|---|---|---|
| PDGM | 495来源限定患者、501扫描 | T1/T1c/T2/FLAIR；A为SynthSeg伪标签；U仅选定序列/标签值 |
| BMSR | 314患者、461扫描 | T1pre/T1post/FLAIR及subtraction；U仅T1post/T1c |
| ISLES | 250患者、250扫描 | DWI/ADC/FLAIR；当前AUR只DWI/ADC，U仅DWI |
| SibBMS | 193患者、372扫描/session | T1w/T2w/FLAIR；当前模板清单只A/序列表征，无U监督 |
| HCP | 1113患者、各有T1w/T2w | SSL TRAIN1001、validation112，非新增独立患者 |
| fastMRI脑 | 5847卷、92590个RSS平面 | FLAIR451卷/7150平面；8213框行涉及997卷，框不是dense U/医生R |
| OASIS | 416患者、436扫描 | Analyze强度/fseg436对同格点，标签只有0/1/2/3，不能代替A32或健康阴性 |

fastMRI脑有5847个不同patient_id元数据值，XML FOR只有4634个不同hash；两者均不是生物学去重证明。SKM-TEA、fastMRI膝、AMOS/SPIDER/TotalSegmentator/KMAR等有其它任务数据，本轮不扩展脑A32/R13本体。SKM-TEA的476条人工组织属性框不能当脑Level R。

182个档案中11个ZIP只读central directory，未提取payload；BMSR/ISLES/SibBMS档案列出的成员没有DICOM。171个tar.gz未作流式内容全审（SKM-TEA157、OASIS14），不能由已展开文件结论声称所有容器内均无原生DICOM。

## 患者、标签与曝光

| 当前角色 | 行数 | 来源限定患者 |
|---|---:|---:|
| 监督TRAIN（排除val_patients） | 3499 | 894 |
| 开发validation | 405 | 101 |
| 历史test | 1056 | 249 |
| SSL TRAIN | 5501 | 1895 |
| SSL validation | 629 | 213 |

监督三角色身份交集0；跨来源生物学去重未证明。PDGM/BMSR/ISLES/HCP没有当前监督或SSL清单外的来源患者；SibBMS8人、OASIS416人未入两份清单，只是候选。fastMRI剩5400个元数据ID未出现在旧447个outline/QC ID集合中，不等于所有历史模型从未曝光。不能把旧test切成2D重新命名独立队列。

原Stage I日志为8889步/320004 crop exposures；选中step8500导出对应306000 exposures。条件sampler回放覆盖全部1895 TRAIN身份，不是完整历史逐患者日志；旧Stage II/III为240000/80016 exposures。此次没有模型训练。

标签数组全量覆盖实际TRAIN2108张独立anatomy图（监督1107+HCP SSL1001），另128张已暴露开发val地图单列；2236/2236成功，历史test dense读取0。当前有效A监督1068图/3420行（39厚层图禁A），32类全部有有效监督体素。

| aseg类 | ignore外存在图/1068 | 有效体素 |
|---|---:|---:|
| 5 | 1063 | 588264 |
| 44 | 1057 | 630313 |
| 18 | 1066 | 1981124 |
| 26 | 1065 | 740213 |
| 58 | 1064 | 764405 |

这排除了“五类在全TRAIN完全无标签”的解释；体素稀少仍需检查采样/优化/类别竞争，不能仅据此确定单一因果。PDGM TRAIN54/359个T1c没有该序列目标值1/4，而FLAIR359/359有目标；这不等于54个健康患者。空u_values、未标注U、原生空标注必须有各自资格语义。

## 已执行的数据与代码修复

180个TRAIN ISLES中，原生177阳性/3空；旧派生142阳性/38空。35张阳性源图被截断成空，源proxy scaling的最大近整数偏差2.3283064365386963e-10。原代码和原native anatomy目标grid逐体素复现180/180；4例不能用写回后的affine替代原目标grid来声称完全回放，这是header精度差别，未改原transform。

新增`labels.discrete_labels`先验证有限实数、近整数绝对容差1e-5及输出范围。整数保持精度，浮点才round，Python整数极值检查避免float16/float32范围比较绕过。概率mask、NaN/Inf及溢出显式拒绝，错误不回显患者数据。整数类别先解码再nearest-neighbour重采样；dataset、外部U评价、fastMRI参考入口复用，连续强度路径不变。

私有新版本`~/outputs/aur_data_audit_20261019/labels/repair_train_v1/`：

- 35张新mask恢复770019体素、459个≥10mm³组件；新数组独立nearest/round逐体素一致、同当前anatomy网格、源和旧mask字节不变。
- 新TRAIN manifest3499行，70条DWI/ADC引用更新；3个原生空DWI仅在新清单作为U未知/R不可用，U-supervised1216→1213。
- 新manifest SHA256：`40feb9b06a56657935a62873a7e52bb57446935b6b107e2a31f7f24fccd24f28`。文件名及逐例内容只在私有目录记录。
- 生产`load_volume`全35例实例0→459，与独立逐例计数一致；A-ignore0→770019，entity文件/图、affine/spacing及A开关不变。A-ignore的变化是修复目标语义的一部分。
- 冻结三源各1个TRAIN身份，6次真实2D/3D crop与collate通过；二维depth1、mm²、默认R资格关闭，三维保持既有语义。无模型forward、GPU或新val/test dense读取。

新版本31.675秒；loader/crop集成39.023秒，158个输入文件hash不变。最终decoder与集成v1快照已逐字节相同；另一次58.092秒的同冻结输入重放通过，35/35转换/最终解码数组hash一致，属于同版本复核，不是新独立医学结果。具体版本和源码hash见RUN.json。已有全零派生无法从自身round恢复，旧`resume`也只会保留坏文件；使用新manifest才获得恢复目标。开发val尚未应用本次修复，不能直接用于新正式选点。回退为选择旧manifest，历史结果/缓存/hash合同始终保留。

## 全量格式、DICOM及来源关联

342889文件前132byte全查，113409 DICOM header全读、无额外隐藏扩展DICM；任意未知裸Dataset或压缩payload不作不存在证明。7174 H5 attrs/shape/XML全覆盖；55044原始/标注NIfTI全读头，另24526派生，共79570。31391个非当前引用legacy派生NIfTI未全审。

全部DICOM为classic单帧scalar MONOCHROME2 MR，enhanced/SEG实际样本0。raw fastMRI112609帧（脑70732、膝41877）全部缺IPP/IOP。800个derived目录对象有几何但UID不合规、重复/不规则位置，无合格规则栈；ImageType ORIGINAL不能单独证明来源。不能将这些对象改名充当原生脑DICOM。

5640卷（脑4468+膝1172）完成H5路径、非空患者hash、shape/depth及零基索引候选关联；XML/DICOM FOR hash全不匹配。112609帧PixelSpacing全部对应reconSpace，76032帧不同于encodedSpace。H5 XML虽有matrix/FOV和patientPosition，却没有逐片position/read_dir/phase_dir/slice_dir/IPP/IOP。两套尺度解释须保留，不能未经上游验证认定任一方是真实患者尺度。

仅既有已曝光brain.csv/raw TRAIN三卷九帧做内容QC：初始严格线性量化3/9通过；同帧观察上下限双端裁剪的事后拟合9/9，不能证明全帧无损、具体生成过程、左右手性或患者坐标。原严格失败保留；未读k-space或显示/保存像素。

当前1mm4960行与SSL6130行的image/anatomy/所引用U header均可用且同格点；原source-grid保留12组image/lesion双form冲突及1A/5U exact grid差异。两份4960行按完整身份唯一配对，但DICOM原帧链字段0。原始2个数值二维SibBMS文件均未入AUR清单，未建立完整原生2D A/U/R验收队列。source PDGM501个4D+文件须先解释非空间维度，不能直接当3D空间卷。

## 初始化、二维R与医生真值

可信Stage I backbone导出160 keys严格迁移；选中step8500完整SSL recovery的176 keys也严格载入，骨干与best export逐位相同，统一StageOne仅4个新增二维tensor未训练。warm-start资格已经核实；旧checkpoint缺完整CUDA/Python/NumPy RNG，不承诺exact resume，也未跑新forward或完成双mode G1。

显式真实二维R资格行仍为0。现有A/U资产足够构建保守的可见范围weak-positive资格，但须按plane/FOV/变换重算，不能把source-instance布尔值广播到所有平面，也不能把未观察到宿主当no-host。详细规则已写设计，生成器与覆盖测量尚未实施。

实际Level R库存在：1297 lesions、165登记患者、2594 orders、2读者账号；labels=0、adjudications=0、releases=0。不是“没有建库”，也不是已经有医生真值。没有联系读者、提交自动答案或发布标签。屏幕左右与患者左右仍须可信映射。

## 验证、审阅、局限

新缺陷先7条RED、修复后35项相关检查；独立代码审阅发现1个P2 dtype边界问题，再6条RED后修复。最终45项相关检查通过（12.19秒）；增量审阅0项，另57项合成边界检查通过。全CPU最终**1528 passed、5 skipped，202.88秒**；5项为nndet环境模块、1条CUDA bf16和3条CUDA恢复用例，本轮未重新运行专用环境/GPU。这次修改在CPU标签边界，不能将旧补测写作本轮新结果。

沙箱内首次全套在DataLoader多进程等待处停滞，394.33秒后中断（377 passed/2 skipped，不算完整通过）；经环境权限批准后在沙箱外完成上述全套。观察命令、源文件hash和证据hash见RUN.json。

独立几何审阅修正3项汇总/声明问题，复核0剩余；代码审阅2轮后0剩余。设计初审0项，后续计划审阅修正1项P2：源overlap必须由源宿主图/计数证明，不能从与nearest回退相同的one-hot目标反推；复核0剩余，未将该生成器写作已实施。新正式训练尚未开始，不推断标签恢复已改善模型A/U/R性能。人标、独立患者队列及原生患者坐标的外部证据不能由代码生成。

私有审计根为`~/outputs/aur_data_audit_20261019/`，目录0700/文件0600；Git仅记录匿名汇总和规则。早期一次读取旧adapter源码的工具输出带出其9个硬编码排除病例标识，已记为输出边界偏差，不重印、不声称全会话匿名。提交产物另做标识扫描；逐患者/帧/UID记录、修复mask、manifest均不入Git。
