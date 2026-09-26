# Level R 读片工具：部署与运维

规格 `docs/superpowers/specs/2026-09-25-level-r-annotation-tooling-design.md`，实施计划 `docs/superpowers/plans/2026-09-26-level-r-annotation-tooling.md`。
所有命令在仓库根目录执行，`python` = `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python`，`D=/data2/congcong/data/FM_data/derived/level_r`。

## 一次性准备
1. 导出图像与病灶：`nice -n 19 python scripts/level_r_export.py --out $D`（165 卷，约 473 MB；已存在 `lesions.json` 会拒跑）。
2. 折表与 pilot 已入库：`data/level_r/folds.json`、`data/level_r/pilot_150.json`。重生成命令见两份脚本头部，脚本拒绝覆盖。
3. 建库：`python scripts/level_r_admin.py init --db $D/level_r.sqlite --lesions $D/lesions.json`。只有 `init` 会新建库；其余子命令、服务和报告脚本只打开已存在的库，`--db` 写错会直接报错，不会留下一个空库。
   `init` 对已有的库可以重跑：只补缺的表（病灶已在就跳过）。2026-09-26 加 `releases` 表之前建的库（例如冒烟库 `level_r_smoke`）要先重跑一次 `init`，否则服务和其余子命令会报 `lacks the Level R tables ['releases']` 拒绝打开。
4. 读者：`python scripts/level_r_admin.py add-reader --db $D/level_r.sqlite --reader-id r1 --role reader --display "读者 1"`（r2 同理；裁定人 `--reader-id adj --role adjudicator`；第三位 `reader` 会被拒绝）。token 只打印一次，记到用户手里，不写进仓库。
5. 顺序：`python scripts/level_r_admin.py order --db $D/level_r.sqlite --reader-id r1 --seed 1 --pilot data/level_r/pilot_150.json`，r2 用 `--seed 2`。

## 起服务
```
setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/level_r_server.py \
    --db $D/level_r.sqlite --data-root $D --pid-file $D/server.pid > $D/server.log 2>&1 &
```
默认只听 `127.0.0.1:8790`。医生访问方式两种：用户在自己机器上 `ssh -L 8790:127.0.0.1:8790 <server>` 后打开 `http://127.0.0.1:8790/?token=<token>`；或明确加 `--bind 0.0.0.0` 走内网地址。停服务：`kill $(cat $D/server.pid)`。sqlite 每次提交即落盘，崩溃不丢数据。

## 每次读片结束
- 导出：`python scripts/level_r_admin.py export --db $D/level_r.sqlite --out $D/export`（每次写进新目录 `$D/export/<时间戳>/`：`labels_<reader>.csv`、`adjudications.csv`、`final_labels.csv`；同一时间戳已存在就拒跑，旧导出不动；两位读者都建好之后才能导出）
- 备份：`python scripts/level_r_admin.py backup --db $D/level_r.sqlite --out $D/backup`（带时间戳，不覆盖）

## pilot 报告
读者读完自己顺序里的 pilot 病灶后，工具停在列表页并提示"pilot 已完成，请等待通知再继续"，不再给下一例。两位都读完后：
`python scripts/level_r_report.py --db $D/level_r.sqlite --pilot data/level_r/pilot_150.json --out docs/verification/$(date +%F)/level_r_pilot.md`
门（spec R7）：全体 raw 的 95% 区间下限 ≥ 0.80 且 0 mm 档 raw ≥ 0.70。门没过先回 v2.6 §3 本体与表单再议，不硬推。
决定继续读全集时逐位放行（只追加一行记录，读者刷新页面即可在同一链接里继续）：
`python scripts/level_r_admin.py release --db $D/level_r.sqlite --reader-id r1`（r2 同理）。

## 封存（全集读完、裁定完之后，一次）
先跑一次导出，`--final` 指向这次导出目录里的文件：
`python scripts/level_r_admin.py seal --final $D/export/<时间戳>/final_labels.csv --folds data/level_r/folds.json --out $D/sealed`
`final_labels.csv` 必须覆盖折表里的全部病灶、每个只出现一次、没有 `pending`，否则拒绝封存、什么都不写。折数取自 `folds.json` 的 `k`。
写 `$D/sealed/labels_fold{k}.csv` 与仓库里的 `data/level_r/sealed_manifest.json`（提交它）。之后开发只用 `anatobind.eval.level_r_labels.load_train_labels(k)`；`load_test_labels(k, unblind=True)` 只在最终评估调用，每次都记到 `$D/sealed/access_log.txt`。

## 医生看不到什么
SynthSeg、任何模型输出、距离档、采集系列、患者号、fastMRI+ 原标签文字、h5 文件名。白名单在 `anatobind/level_r/blind.py`，服务端每个 JSON 都过 `assert_blind`。
