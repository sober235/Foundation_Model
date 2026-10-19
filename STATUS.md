# STATUS：2026-10-19（TRAIN U背景配额切线完成；该候选不获支持）

当前阶段：按已批准方案完成原六名TRAIN患者/十二裁块的配对背景采点实验，执行代码73aa2c7。只在诊断U点集把boundary_share0.2改为0，保留16384点、focus_share0.5和原匹配，测量原生值锚定的float32共享掩膜头切线。ISLES主面板8/8裁块×draw的combined单位方向差global−control均为正（均值+1.438208），未采到FP背景的局部抑制更弱，四裁块跨draw均值也4/4同向。当前结果不支持把该改动推进为修复或据此加训；正式采样器/损失/权重未改，没有val/test前向。默认legacy_max、完整405val FAIL及A/R缺口保留。

最新报告 `docs/verification/2026-10-18/aur_u_background_allocation/RESULTS.md`；同目录RUN.json、aggregate_anonymous.json、supplement_anonymous.json、numeric_audit.json。目录日期来自2026-10-18冻结方案，实际执行于2026-10-19。上一轮U监督报告仍为 `docs/verification/2026-10-18/aur_u_train_supervision/RESULTS.md`。更早分数排序上界继续限制既有最终标量的单调校准拟合，不排除新特征、改变排序或更好掩膜。

## 0. 最新背景配额对照与复用

- 主面板排除两臂采样点并集，只测原匹配query的官方FP有效真背景；先query内合并组件，再按crop/draw求均值。ISLES56条query×draw、8/8crop×draw有支持，control−0.859799→global+0.578410；原始combined及U-only单位方向差也8/8不利。仅局部切线，不是FP数量变化或完整网络/历史训练梯度。
- ISLES远处背景采点38536→64225，但已匹配FP真背景唯一覆盖体素×draw3085→1729；PDGM覆盖33→52而方向仍不利（4/8crop×draw有支持）。BMSR已匹配FP面板无支持，未匹配query路径仍独立保留，不能当成零效果。
- ISLES分配目标logit方向差均值+1.224134；A损失方向差均值−0.004193但5/8较好、3/8较差。PDGM/BMSR的A差均值为正，即较少下降；两臂combined单位方向下的全部A总损失方向仍为负，不是实测A Dice损害或修复。原固定query R logits、十二裁块完整模型状态完全相同，这是受限路径控制而非R性能提升。
- 重新按干预点官方匹配后5/106目标×draw换query（均ISLES，5/86）；不是前轮两个原draw之间的5/53，主分析保持原匹配。106条两臂均无有效正点缺失，无容量溢出。
- 完整CPU1345passed/2skipped180.61s；独立任务/整分支代码审阅0项问题。实际A800物理7两次空闲快照/兼容性通过后运行161.126s、退出码0，最大allocated2.510GiB/reserved3.977GiB、RSS2.860GiB。先前候选物理5在启动前已忙，保护检查在模型/影像加载前停止；既有任务保留。
- 纯CPU完整回放15.095s、退出码0，27成员字节不变，显式阻止模型/影像/checkpoint加载。独立脚本核对12crop/24draw/680ROI，69297项检查通过；未独立重新生成密集特征均值，其来源另由runner密集流式约简和输入/状态控制核验。TRAIN资格/完整源承诺由controller固定脚本核验，不混称独立溯源。
- 私有完整缓存 `/home/congcongliu/outputs/aur_u_background_allocation_20261018/run_73aa2c7`，只用隐藏CUDA的 `scripts/aur_diagnose_u_background.py --out <该目录> --replay-only` 复算，不重复GPU测量。128项原依赖、checkpoint和旧29成员参考存档不变；新医学派生数组/逐例记录留在0700/0600目录，不入Git。当前新增公开文件和实际固定stdout另行扫描，不能扩展为历史会话匿名保证。

## 0.1 上一轮TRAIN U监督诊断的读法与复用

- 来源异质：BMSR的6个FP在两次采点均来自未匹配query；ISLES100个FP对应176匹配/24未匹配query的FP组件×采样记录；PDGM7个FP对应6/8。不能把ISLES主导的全局模式直接概括到BMSR。
- 106个目标×采样均有有效正点，未遇query容量溢出；5/53裁块目标换了query（都在ISLES）。这些是新固定TRAIN裁块/CPU点draw，非历史分布式训练流或参数梯度重放。
- 私有缓存 `/home/congcongliu/outputs/aur_u_train_supervision_20261018/run_1a7de61` 已完成；只需隐藏CUDA运行 `scripts/aur_diagnose_u_supervision.py --out <该目录> --replay-only` 做只读完整复算。当前source/spec/原始输入/选择/权重承诺必须一致；报告提交导致HEAD前进但代码闭包不变可复用，不要重跑GPU补已完成的测量。
- 模型状态锚点5fbb0d6f9513fdebf66831033c6a8d32710a05ce2ec900656c062a61546e482f，12裁块前后均一致；原生产Python文件未改。启动至完成标记约111.467秒上界，allocated峰值2.609GiB、reserved3.977GiB，实际CPU回放29.87秒。不是推理速度比较。
- 两次子任务工具输出误带私有标识（上下文选择记录、审计清单路径键），第二审计任务已停止。新/改文件、匿名汇总与实际runner日志另行核验；不能把这些检查写成整个会话终端匿名。私有医学派生数组与逐query/组件记录继续留在0700目录，不入Git。

## 1. 三个目标及当前裁决

**A32解剖掩膜、统一异常U实例掩膜、每个U的R13分侧主宿主+no-host（14概率）不变。** S仅为内部序列上下文。默认解码仍为legacy_max；可选query_cc_v1只保留为可复现实验，不能当作已通过的模型修复。

| 目标 | 既有完整val结果（本轮未重跑） | 状态 |
|---|---|---|
| A | 394个A监督行的13-host Dice0.803457；405行保存分数逐项完全不变 | val宏门通过；五类缺失仍未解决，最近历史test0.7988仍不过 |
| U | BMSR/ISLES/PDGM-FLAIR检出2→78、4→60、27→45；FP/scan变为17.0、540.1、4.7317 | 三源均超S7预算，也均低于参考敏感度减0.05；该固定profile不采用 |
| R | 716个controlled实例的共享保存字段完全不变；ABA0.568436对B00.698324 | controlled门仍失败，未修复R |

前轮排序报告 `docs/verification/2026-10-18/aur_score_order/RESULTS.md`，同目录 `RUN.json`、`aggregate_anonymous.json`、`manifest_exposure_anonymous.json` 保存命令/哈希/匿名结果与患者清单资格审计。置信度测量仍见 `docs/verification/2026-10-17/aur_residual_confidence/RESULTS.md`；完整405行对照仍见 `docs/verification/2026-10-17/aur_query_cc/RESULTS.md`。那六例为已暴露开发val，与本轮固定TRAIN六例不同；均仅机制诊断。A/R采用伪参照，整体NOT_EVIDENCE。没有Level R人标、独立校准/总体过门或临床成功主张。

原六例固定阈值下，parent→子raw均值的事后过滤结果：BMSR命中2/13不变、FP16→23；ISLES命中11/40→10/40、FP1511→171；PDGM命中2/2不变、FP6→0。三种分数校准不同，反事实保留全部原mask/NMS/ownership，仅过滤后重匹配，不是新的正式解码性能结果。

本轮使用同六例S7预算0.5/1.5/0 FP/scan（总FP≤1/3/0），参考命中8/13、18/40、2/2；不能混用完整405预算0.8780/1.6/0.3659。三个分数上界相同不代表全部排序相同；单调最终标量映射不增加可保留集合，但此结论不覆盖体素变换或新特征。S7native/AUR1mm差异仍在，不作新正式门裁决。

## 2. 已完成的实现与控制

- 最新排序分析代码88c38b6，38个冻结依赖保持一致；原固定工作点和A/U/R存档控制逐项复现，17,350个集合全部按原Hungarian+ignored规则重匹配，不保存最大化阈值。无模型前向，存档不变不等于新A/R质量提升。
- 前轮置信度测量代码6b26137，原推理/模型/训练36个依赖文件与冻结1874源码一致。7528个子实例测量parent/raw/fused三项分数；原U行、GT、解码统计、legacy检测、六例A保存分数、55个R共享字段及query/window来源精确复现。两遍当前A/U图和上下文精确相同，历史未保存图不作跨轮声明。隐藏GPU的完整缓存恢复核验通过，未加载模型。
- 冻结实现代码1874af2，checkpoint SHA256 `38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde`；实际best选自Stage III step3000/72,000crops，80,016crops为已完成训练预算，权重本轮未改。
- query_cc_v1：独立窗口/query的26CC，presence>0.5、mask>=0.3、共享10mm³floor、原mask IoU>=0.5贪心NMS、按固定优先级归属体素后再拆残余组件，子组件继承父分数；无GT挑选、无候选上限、无回填。detector来源和重裁块后实际R query来源分别记录。设计与计划在 `docs/superpowers/{specs,plans}/2026-10-17-aur-query-cc-decoder*`。
- 405/405保存A分数严格相等；716/716controlled R共享字段严格相等。旧缓存未保存A体素图、controlled query/window来源，不能声称这些未保存对象也已逐项验证。
- 旧端到端缓存先匹配再过滤分数的问题已在本对照中统一为先固定阈值过滤、后一对一bbox IoU>=0.1匹配；2个保留检测恢复GT关联，旧匹配池38→40，预测未改变。原始缓存保留。
- 716个controlled合格GT的正确端到端宿主27→128；共同匹配的40个GT正确数27→26（0rescue、1harm）。新增成功主要来自检测覆盖，不能说R头质量提高。U共718个非忽略GT，有2个没有controlled记录；分母分开报告。
- 上一轮TRAIN U监督诊断完整CPU测试1244 passed、2 skipped，171.70秒；新四模块集成86项通过，独立代码/数值审计0项未解决。隐藏GPU、nice19、线程2，沙箱外解决DataLoader多进程环境限制。前轮排序分析的1170/2（163.48秒、19项针对性）、更早1151与1111项及A/R审计保留为对应版本历史，不能混写为本轮新性能结果。
- bf2a924历史CPU跳过项曾另行补测：nndet环境坐标模块2 passed（2.39秒），nvgen CUDA混合精度单测1 passed（19.06秒），均0失败/0跳过。历史CPU原计数不改；这是一个模块加一个单测，共3条用例，不是本轮新补测。nnDet6条第三方依赖警告保留；合成数据测试不构成新的A/U/R性能结果。证据 `docs/verification/2026-10-18/aur_score_order/supplemental_tests.json`。

## 3. 误报与 A/R 机制线索

- 数值源分数固定BMSR0.70、ISLES0.60、PDGM0.75；raw父组件均值与legacy融合图均值校准不同。本轮比较不是等观测FP，也没有搜索阈值；不能由此否定所有query保留方案。
- 完整val的既有事后形态分析：ISLES10,802个FP中，84.06%子组件保留父体积<=10%，89.98%来自多子组件父组件，子体积中位数36mm³；该完整集当时没有保存子概率。本轮六例直接测量：ISLES原1511个FP的parent/raw均值为0.63888/0.46675，1350个raw低于0.60；但BMSR原16个FP中14个保留比例>0.5且raw中位数0.71872，自身仍过0.70。直接换raw不能统一修复误报；没有把六例统计外推为完整val统计。
- A标签覆盖审计：8个TRAIN行、16裁块、32点采样；共享采样42.282%点落在A忽略区，uniform为1.350%。密集存在但采样无正点时，实际mask损失推动空预测；18/54在病灶裁块的正点均值及零正点频次相同，尚不能据此建立单侧18缺失的因果解释。见 `a_supervision_audit/REPORT.md`。
- R粗投影与dense投影独立，无直接A/U掩膜或一致性监督；Stage II粗投影与seed0初始化逐位相同，Stage III后改变。共享coarse像素金字塔接受A/U训练，不能称整个粗分支未训练。CPU权重/config/历史源码验证已由root和独立审阅者重跑。见 `r_coarse_supervision/REPORT.md`。

## 4. 下一项模型工作

1. **背景配额对照也已完成，保留原采样，不据此加训**：当前ring0.2→global0候选在固定ISLES主面板的局部方向不获支持；不要在这些裁块上挑选中间配额或直接追加训练。下一项U干预须重新冻结要区分的空间权重/表示/优化机制及A/U/R回退检查；BMSR未匹配query路径与A/R独立任务仍保留。清单U-supervised TRAIN761来源限定患者/1216行，val86/143，历史test212/354；基模型已见TRAIN，未建立新独立评价队列。当前实验不证明U优先级全局最优。默认legacy_max和完整405行FAIL不变。
2. **R单独做固定token/query/crop/权重的几何来源对照**：仅将预测coarse A/U几何换为同一F1网格上的预测dense A/U几何，预先固定池化、valid、gate和坐标规则，记录逐特征和宿主logit变化。联合A/U替换不能区分二者贡献；冻结R头的分布变化也限制阴性结果解释。该实验尚未执行。
3. **A用同一冻结TRAIN裁块审计全部32query的dense概率和胜出类别margin**，分开检查正点遗漏、身份/优化塌缩及类别竞争，再设计单一采样或损失变量。不要仅据18/54相同的正点均值及零点频次直接跳到改loss。

这些是下一步诊断/实验，不是已完成的修复。三个模型目标始终优先；N7、DICOM/独立2D、网页/API/URL及Level R实际读者工作仍按既有顺序处理。

## 5. 资源与别重做

- 新排序分析仅CPU：17,350集合、wall55.38秒、脚本49.68秒、RSS847320KiB，CUDA隐藏/nice19/线程1，600秒内外限时；不重复拟合仅改变同一标量刻度的模型来期待改善可达排序上界。已规划SibBMS10人子集含7TRAIN/3历史test，不能包装成新评价；来源限定清单互斥不代表已证明跨来源生物学去重。
- 新六例回放已在空闲A100物理卡3完成：成功wall186.79秒、行时间合计175.66秒、峰值CUDA allocated9.0176GiB，CPU nice19/数值线程1/decoder4。首次外部启动器参数转发失败发生在模型加载前，2.57秒；修正入口并核验help后重启，日志保留。不要重新跑这六例来补已完成的数值测量；相同冻结合同可直接验证恢复。
- 完整val4302自然窗口，累计冻结单行推理4051.38秒，峰值已记录单行CUDA分配9.017GiB；并行墙钟跨度约4282.42秒含调度/空闲，不能与缓存legacy时间作速度对比。六例NMS加速前后全部保存结果相同，实测507.78→406.60秒为共享主机观测。
- 先检查分片的source/sequence/U监督分布。排序后步长2使全部20个DWI落一端；本次保留已完成odd202行，验证原始全部输入哈希后复用even68行，最终以1/2、0/4、2/4三完整分片合并202+102+101=405，原0/2partial不入结果。见 `scheduling/repartition.json`。
- 恢复必须保持code/checkpoint/所有原始输入/规则/参考哈希；重新分片还必须核对未完成行的原冻结输入哈希。逐例/逐query表、影像与Level R盲读材料只留本地private目录，不能入git。
- 数据只读/data2/congcong/data/FM_data；nvgen绝对路径、PYTHONNOUSERSITE=1；CPU测试隐藏GPU。GPU按用户常设授权，先查实时负载/显存/活跃任务及调度可见性，选兼容空卡并保留既有工作负载。
