# 双模式评价接口与二维保存证据诊断：首批交付

已落实[批准路线](../../../superpowers/specs/2026-10-21-anatobind-core-goal-and-evidence-route-design.md)的第一步。实现不改变原始 G1、患者角色、默认解码或医学质量裁决。**接口可用性通过合成端到端测试，真实统一监督模型的质量仍未评价。**

## 实现结果

- `model_from_unified_export` 严格载入 format2 II/III 监督产物，验证真实完成证书、数据身份、架构、完整有限权重及类型转换溢出，再转移设备。format1 保持 pilot-only；完成阶段不等于认证质量。
- `scripts/aur_unified_eval.py` 冻结 development_validation 角色、源行和按 header 选定的完整中央平面，分别评价两模式。A32/13-host 支持与 Dice、U 固定阈值检出与误报、受控 R/端到端 R 和对应 B0 均单独记录；未知参照为空，host0 有效。
- 二维 U 按已解码原病灶二值图作8连通、至少1mm²；三维保留26连通/10mm³。二维 R 独立要求可靠源实例与同平面宿主可见性。FP/image 和 FP/scan、面积和体积不混用，完整观察不继承 TRAIN 的64-query容量限制。
- 历史源码使用显式 evaluation 校验路径，继续检查实际源字节、角色引用、原合格源码和记录的环境。默认训练仍检查 live 代码。实际 final_v3 的5项源码与65eb7df逐字节相同，但其中1项缺历史精确副本；新私有归档明确是审计时从 Git 重建，不改原合同。
- 实际合同复核暴露了新校验对历史父合同环境字段要求过强的问题。新增回归先复现，再修复为叶合同必须记录环境，父合同有记录则校验；父合同未记录的历史字段不补造。真实合同完整复核最终 PASS，198.829秒，405验证行/3499训练行，合同未改变，模型前向0。

## 验证和审阅

正式 loader 68项针对性检查通过；评价/数据合同/平面几何等149项通过；父合同修复56项通过。计数有重叠，不相加。最终沙箱外 CPU 全套为 **1768 passed、5 skipped，282.65秒，退出0**。跳过为 nndet 专用环境模块1项、CUDA bf16单测1项、CUDA优化器/RNG检查3项；这些未被本次 CPU 回归覆盖。

沙箱内两次全套在既有并行裁块检查处停住，未作为通过证据，后通过各自工具会话终止（exit130）。按项目既有说明在沙箱外执行最终全套成功。没有为通过修改该检查。

每任务独立审阅及整分支终审完成，首个合同兼容发现已修复，剩余可行动发现0。终审覆盖 f3eb35f..792e93a；完整测试覆盖生产代码 b4244dd，后续文档提交不改变运行代码。原[二维诊断](../aur_g1_diagnosis/RESULTS.md)的独立证据检查也通过；它没有复测3D大缓存或建立模型因果解释。

准确执行记录、源码闭包及日志哈希见 [RUN.json](RUN.json)。测试采用 nvgen、隐藏CUDA、nice19、数值线程2，最终命令主体为：

```sh
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -rs -p no:cacheprovider
```

## 使用与恢复

将 `<completed-supervised-export>` 换成以后实际完成的统一 II/III 导出。当前没有这样的真实模型，不能拿 Stage I 或旧非统一模型冒充：

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/congcongliu/anaconda3/envs/nvgen/bin/python scripts/aur_unified_eval.py --contract /home/congcongliu/outputs/aur_qualified_training_20261019/data_contract/final_v3/contract.json --qualified-source /home/congcongliu/outputs/aur_qualified_sources_20261021/65eb7df --export <completed-supervised-export> --out <new-private-output> --device cpu
```

示例是待用模板，未执行真实队列推理，不应复制尖括号占位符运行。目标目录必须是许可私有 home 下、Git 外的新目录；失败保留计划且无成功标记，修正输入后使用新目录。数据与旧运行无破坏性迁移；如需回退，只选择旧代码/接口版本，保留私有证据，不改旧数据。

## 尚未完成

原 G1 仍为2D FAIL/3D PASS；II/III仍未启动，最终两模式 A/U/R、原生DICOM几何、独立患者评价和医生 Level R 均未完成。医生标签仍0。下一有限对照使用实际step0和selected1792、固定二维观察，另行登记与审阅，不能把本接口终审当作该对照已获科学验证或阶段放行。
