# 数据修复后的统一2D/3D训练实施计划

**Goal:** 用已验证的新TRAIN目标和可信Stage I权重，完成真实两维数据资格、纯图像SSL、正式训练/双mode G1闭环；原生DICOM与医生R保留各自验收。

**Spec:** [数据驱动设计](../specs/2026-10-19-aur-data-grounded-gap-closure.md)。设计独立审阅0项；本计划区分已执行Task 1与后续任务，不声称正式训练已经完成。

**Baseline:** 数据/标签修复代码`62650153c0c48d55d16d8a2fd166a310d2810edc`；[审计及验证证据](../../verification/2026-10-19/aur_data_audit/RESULTS.md)。

**Constraints:** 脑MRI、一次一个序列、同一模型/单checkpoint；共享主体+维度stem/up，A32/U/R；R为13宿主加`NO_HOST=13`，共14类，host_index0有效，unknown不是no-host。`/data2/congcong/data/FM_data`只读，派生/清单/逐例记录在home私有目录0700/0600；不读冷备份，不重写历史缓存/hash，不把历史test变成新评价。Python使用nvgen绝对路径、CPU nice19/数值线程2、export workers≤4；GPU先检查可见性、调度/利用率/显存/活跃进程并选择兼容空卡，常设授权不重复询问。

**Validation prefix:** `env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python`。下文`<CPU>`指此前缀；不是可直接输入shell的变量。医学CLI错误只输出固定匿名类别，不回显参数值。

## 执行账本（2026-10-21 UTC）

执行代码`65eb7df`，完整证据、实际新增文件/接口及验证以[本轮报告](../../verification/2026-10-20/aur_qualified_training/RESULTS.md)和RUN.json为准。下方各任务保留原验收条目；该账本区分代码交付、实际有限验证与未完成的科学验收。

| 任务 | 当前实际状态 |
|---|---|
| 1 标签修复 | 已完成，沿用原冻结证据 |
| 2 数据资格 | 已完成当前主配方合同及全量源/输出校验；TRAIN819份补充数组QC、val20 ISLES全体与另3份修复已完成；候选FLAIR/OASIS扩充未纳入本轮 |
| 3 2D弱R | 已完成代码/回归/独立审阅；真实TRAIN36观测、39组件中26合格 |
| 4 纯图像SSL | 已完成loader、strict迁移、双mode梯度/恢复检查及176+4实际warm-start；新增迁移实现位于ssl/unified_init.py |
| 5 编排/G1 | 训练/G1代码、完整恢复/完成证书、判门测试及审阅完成；实际G1为2D FAIL / 3D PASS。最终format2/新validation角色/双mode A/U/R评价接口及held-out二维R资格仍未完成 |
| 6 资源与正式run | 全尺寸I/II/III各8步probe及CUDA恢复完成；正式Stage I完成2048步、每mode1024曝光，best step1792。2D G1增益0.043545<0.05，formal_pass_v2于03:12:50 UTC按登记规则停止，II/III未启动，不追加训练或改门槛 |
| 7 外部补证 | 5640卷候选映射hash复核，1297病灶Level R工作副本准备并核验；可核原生患者几何、医生实际标签及R0/R1验收仍未完成 |

最终CPU1699 passed/5 skipped/243.92秒。原数据/历史缓存及失败记录不覆盖；医学数组、身份/观测清单与读者数据库留在私有目录。

## Task 1：离散标签修复及实际数据纵向闭环（已完成）

**Files:** `anatobind/aur/{labels,resample,dataset,eval,fastmri}.py`、对应5测试模块、`AGENTS.md`。

**Interfaces:** `discrete_labels(values,dtype=int16)`验证有限近整数与范围；整数不损精度，浮点容差1e-5；整数目标先decode再nearest。

- [x] 近整数前景消失、非法概率/非有限/溢出的7条RED；独立review dtype边界另6条RED。
- [x] 修复所有当前AUR输入标签转换边界，45针对测试通过；独立review2轮后0项、57个边界控制通过。
- [x] 新私有TRAIN清单3499行，35新mask、70条引用恢复、3个原生空DWI关闭U/R资格；源/旧数据不变。
- [x] 实际loader恢复459实例/770019 A-ignore，三源真实depth1/3D/collate通过；同冻结输入最终代码重放不是新独立医学结果。
- [x] 完整CPU1528 passed/5 skipped/202.88秒，沙箱外完成；未把CUDA/nndet旧补测记成本轮执行。

后续监督输入明确使用`/home/congcongliu/outputs/aur_data_audit_20261019/labels/repair_train_v1/samples_train_repaired_v1_private.json`，SHA256 `40feb9b06a56657935a62873a7e52bb57446935b6b107e2a31f7f24fccd24f28`。不能把代码修复等同于旧全零派生自动恢复。

## Task 2：正式数据合同、患者角色与开发validation资格

**Files:** Create `anatobind/aur/data_contract.py`, `scripts/aur_qualify_training_data.py`, `tests/test_aur_data_contract.py`；private配置/ledger不入Git。

**Interfaces:** 版本`aur_data_contract_v1`消费新TRAIN清单、原生/当前row linkage、SSL清单、val患者表和审计hash；输出私有`train_supervised.json`、`train_ssl.json`、`validation.json`、资格原因/来源hash、匿名coverage及完成承诺。路径允许只读数据根以及有完整来源合同的私有修复根；禁止Git内医学输出及未经批准外部来源。

- [ ] RED：patient group跨split、val混TRAIN、同一患者序列/切片拆分、source-empty疾病mask被当健康、输出路径/源hash变化、损坏label和无效物理单位均拒绝。整数解码已合格不再改容差。
- [ ] 同一源身份沿用旧分区；metadata/source-qualified关联与生物学去重状态分开。每个mode都沿用患者组，观测数量不改变身份权重。
- [ ] 对已暴露开发val，先冻结与TRAIN相同的解码/unknown政策，再做原生→派生标签资格检查及必要的新版本修复；不看模型预测，不调类别/阈值。原20 ISLES val患者作为固定全体，不按结果挑选；记录新增数据QC曝光，历史test dense继续封存。
- [ ] 当前3499 TRAIN保留894身份，U有效行1213；新validation必须单列分母、源阳性/原生空/转换丢失、A/U/R资格。未完成validation资格不释放正式best/G1。
- [ ] Stage I只消费image及几何/患者/序列元数据；SSL5501 TRAIN的标签路径不进入loader。ISLES TRAIN180 FLAIR作为单独候选清单，先做序列/几何/QC，不默认扩充本轮主配方；OASIS/fastMRI等后续同理。
- [ ] `<CPU> -m pytest tests/test_aur_data_contract.py tests/test_aur_resample.py tests/test_aur_dataset.py -q -p no:cacheprovider`；然后真实private CLI全量资格审计，输出身份交集0/源hash不变与匿名分母。

## Task 3：观察级二维weak-R资格

**Files:** Create `anatobind/aur/planar_relations.py`, `tests/test_aur_planar_relations.py`；Modify `anatobind/aur/unified_data.py`、`tests/test_aur_unified_data.py`。

**Interfaces:** `qualify_planar_weak_relations(source_instances, source_host_map, transformed_plane, valid, a_ignore, source_instance_ids, pixel_area_mm2)`返回对齐当前组件的`r_eligible/host/probs/negatives/provenance`。`source_host_map`是与完整源instance数组同格点的13宿主类别图，可由未被A-ignore抹除的源entity图解码获得；不是`vol['hosts']`的目标分布。现有`host_targets`将真实重叠与nearest回退都编码为one-hot，不能据其probs反推有重叠；本函数须直接统计源instance与source_host_map正重叠，或消费含源图hash/分母/各host体素数的等价缓存。输出资格只对当前plane/FOV/变换有效，不能保存成对所有平面生效的volume布尔。

`transformed_plane`明确含最近邻同变换后的instance图、host-map和组件→源instance映射，三者与valid/a_ignore同格点；host-map沿用`labels.host_map`约定0=非宿主、1..13=宿主，转换为R目标时减1，R的host_index0仍有效。不能把此类别图的0当R的no-host13。

- [ ] RED：host0有效；构造真实重叠与nearest-only两例，其legacy one-hot targets相同而资格必须不同；另测plane外/并列/no-host/不可见/未知U拒绝、source soft分布含plane外host不继承、U与`~a_ignore`不能被错误相交成空。
- [ ] 实现设计中的双条件：源3D唯一重叠主host，plane唯一重叠主host相同；另需plane宿主在valid且A-ignore外可见。源26CC≥10mm³、plane8CC≥1mm²，裁剪/旋转后重验，身份经组件拆分仍追溯。
- [ ] 合格目标one-hot、`geometry_weak2d_v1`、固定weak权重1.0，沿用现R归约；不调权，不生成医生标签/no-host真值。灰区仍关闭R。
- [ ] 接入监督crop和collate，不改默认未配置二维R=false及旧3D路径；query超64显式拒绝/报告，不截断。source-r兼容入口不得绕过观测资格。
- [ ] 冻结TRAIN患者/plane列表后测coverage，按source和mode报告可见U、合格weakR、关闭原因；保存私有映射，匿名汇总入报告。
- [ ] `<CPU> -m pytest tests/test_aur_planar_relations.py tests/test_aur_unified_data.py tests/test_aur_unified_training.py -q -p no:cacheprovider`，真实TRAIN loader检查不读val/test。

## Task 4：Stage I纯图像双维路径及可信warm-start

**Files:** Create `anatobind/aur/ssl/unified_data.py`, `tests/test_aur_ssl_unified_data.py`；Modify `anatobind/aur/ssl/checkpoint.py`, `anatobind/aur/unified.py`及对应测试。

**Interfaces:** 图像loader只接受image/affine/患者组/seq/valid，双mode SSL views保留物理坐标与mode特异patch/block mask。消费已验证step8500完整StageOne176-key状态，新增4个二维adapter tensor固定seed20261019初始化；输出完整迁移/初始化lineage，不是旧optimizer resume。

- [ ] RED：把A/U文件访问stub设为抛错，两个mode仍能完成SSL crop/forward；无lesion-centred plane选择。独立二维归一化基于完整原plane，不由已clip体积反推。
- [ ] 旧完整SSL各176 keys逐位继承，4个新keys显式列出来源及种子；拒非预期missing/extra/shape/dtype。160-key backbone-only作为另一显式policy，不静默混用。
- [ ] 明确新optimizer/scheduler从0开始，历史step8500/306000 exposures仅lineage；新2D/3D曝光从0，禁止宣称历史RNG完整恢复。
- [ ] 两mode对共享主体产生梯度，非活动适配器grad=None且不被weight decay；无标签数据可训练。CPU恢复精确，CUDA记录原后端/容差限制。
- [ ] `<CPU> -m pytest tests/test_aur_ssl_unified_data.py tests/test_aur_ssl_train.py tests/test_aur_unified_model.py -q -p no:cacheprovider`；现有checkpoint恢复测试位于`test_aur_ssl_train.py`，新测试不装作已运行。

## Task 5：双mode G1与正式训练编排

**Files:** Create `anatobind/aur/unified_run.py`, `anatobind/aur/ssl/unified_eval.py`, `scripts/aur_unified_train.py`, `scripts/aur_unified_ssl_eval.py`, `tests/test_aur_unified_run.py`, `tests/test_aur_unified_ssl_eval.py`；复用现有`unified_train.py`、`ssl/eval.py`及`gates.py`的已验证机制，不把pilot flag删除当完成。

**Interfaces:** 显式完整model/SSL配置、阶段预算、mode/source/patient采样、私有数据合同、初始化、恢复、周期验证及best选择。支持单GPU+梯度累积的首个正式配置，DDP不作为首次运行隐式要求；若以后启用须单独验收全状态/采样恢复。

- [ ] RED：不完整配方、数据/初始化/后端hash变化、mode丢曝光、阶段越门、错误恢复计数、validation患者进入TRAIN、private path错误均拒绝。
- [ ] G1冻结image-only表征，两个mode各自与同配置随机模型对照；probe seeds0/1/2，每mode校准96、validation48、lesion48，按患者bootstrap且两臂共享冻结裁块，按mode/source/host单列。二维分层用面积的等效圆直径`2*sqrt(area/pi)`，<5/5–10/>10mm；三维沿用体积等效球直径，不能把二维面积直接当体积。
- [ ] 将`ssl/eval.py`既有PROPOSED工程门显式冻结到两个mode配置：host macro SSL−random≥0.05、各seed患者bootstrap差值下界的最小值>0、lesion AUC_SSL≥AUC_random−0.02。两mode分别判门，NOT_EVIDENCE。新verdict要求两臂AUC有限且均有正负样本；缺数据/非有限状态为INSUFFICIENT，不沿用旧实现NaN也pass的逻辑。每host/size-bin支持数及缺失单列，不伪造分母或从结果反选门。
- [ ] Stage I→II须两个mode G1都给出完整资格/判门；II→III沿用既有冻结协议。最终按2D/3D×A/U/R验证，R伪参照NOT_EVIDENCE，独立医生门另报。
- [ ] checkpoint原子新建、完整optimizer/scheduler/RNG/采样位置/曝光/数值环境；best/last指针带文件hash，不覆盖历史文件。安全加载只针对可信本项目产物。
- [ ] `<CPU> -m pytest tests/test_aur_unified_run.py tests/test_aur_unified_ssl_eval.py tests/test_aur_unified_training.py -q -p no:cacheprovider`，合成恢复/隔离tests后再做有限TRAIN工程运行。

## Task 6：全尺寸预算probe、冻结正式run并推进阶段

**Files:** 新私有`full_model_probe_v1/`、`formal_run_v1/run_config.json`/`lineage.json`/`data_contract.json`，公开匿名预算/阶段报告；CLI使用Task5入口。

- [ ] 启动前独立读GPU利用率/显存/活跃compute进程及可见性/调度信息；选兼容空卡，保留既有任务。记录实际卡、后端和源/配置hash。
- [ ] 工程probe固定TRAIN seed20261019、完整已核model配置；I/II/III各8步，mode交替各4步，batch1、3D128×160×160/二维1×160×160，总wall上限1800秒。它只验证内存/梯度/恢复；II/III probe不宣称正式跨门或性能。
- [ ] 记录每mode峰值allocated/reserved、step时间、各loss、gradient有限、曝光与源文件不变；先做一次断点恢复，不能承诺CUDA未来参数逐位相同。
- [ ] 据实测冻结正式steps/LR/warmup/累积/验证周期/预计wall/最大预算到配置并审阅。原batch24/240k等仅是历史，不是默认新预算；预算尚未测量时保持formal status NOT READY，禁止启动无上限训练。
- [ ] 数据/SSL/G1/预算合同通过后开始正式Stage I；达到冻结门再转II/III。失败保留权重、失败与回退说明，不自动换loss、改阈值或无限加训。
- [ ] 定期报告mode×A/U/R与源分母，五个A重点类不被宏均值掩盖；医生R和独立患者数据仍不足则相应结论继续未完成。

## Task 7：原生DICOM及医生R的独立补证

**Files:** 私有来源关联/标注工作空间；复用`anatobind/aur/dicom.py`、`anatobind/level_r/{store,export,schema}.py`和现有脚本，不新增重复标注库。

- [ ] 冻结已有5640卷H5–DICOM候选映射，保留SOP/FOR hash、索引、两套spacing及代表性内容检查。它不是原患者affine；保留原strict3/9与事后9/9两份结果。
- [ ] 先查可验证的本地上游/档案资料；合法UID、IPP/IOP、非空间维度及原帧来源完整后，另建私有DICOM验收集和回原帧检查。不能静默改800份对象UID或虚构IPP/IOP。
- [ ] 指向已发现的1297-lesion真实Level R待标库，在新私有工作空间复用，保留原只读源；labels=0时出EMPTY状态，不生成自动医生答案。准备读者材料可独立验收；联系/发送材料需明确消息授权，当前未发送。
- [ ] 医生实际提交/裁定/封存后才验Gate R0/R1。屏幕侧别与患者侧别映射、acceptable set及独立评价患者须明确，不能用几何伪标签或新切片替代。

## 收尾与复核

每任务先看starting diff和实际配置/测试入口，独立交付后提交；中风险代码/数据变化做fresh-context审阅，实质发现修复或证据性记录，自动循环≤3轮。最终运行项目完整CPU套件；只有改动影响相应路径或出现新问题才补跑CUDA/专用环境，不重做已有GPU诊断。新医学数组和清单不入Git；公开报告扫描标识、检查所有Markdown本地链接和JSON，再按用户已要求的新分支方式push，远端main不直接推送。

完成本计划不等同于六项质量门全部过关：必须分别给出实际测量、未通过项和外部依赖。网站/API/URL继续后置。
