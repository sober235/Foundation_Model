# Level R 冒烟部署记录（2026-09-26）

对应实施计划 `docs/superpowers/plans/2026-09-26-level-r-annotation-tooling.md` Task 12 的 Step 3（真实导出）与 Step 4（冒烟部署）。冒烟库与正式库分离：
冒烟库 `S=/data2/congcong/data/FM_data/derived/level_r_smoke`，只读复用真实导出 `D=/data2/congcong/data/FM_data/derived/level_r`
的 `volumes/`。冒烟服务端口 8791，与正式服务的 8790 分离。

**方法学说明（不改变命令语义）：**
1. 服务启动后先查看 server.log 再发请求。
2. 部分核对命令拆成独立命令逐条执行（变量以字面值代入），数值不变；curl 检查一节里的 `$L`/`$V` 分别用字面值
   830 / b3e1f6f0 代入。除此之外命令与输出均为逐条真实执行的原样记录，未做拼接或删减。
3. 读者/裁定的 token 只在 `add-reader` 输出里出现一次，已 `tee` 到 `$S/tokens.txt`（不在仓库里、不提交）。
   本文档里所有 token 一律替换成 `<token>`。

## Step 3 真实导出

真实导出（165 卷、1297 病灶），`D=/data2/congcong/data/FM_data/derived/level_r`。

```
$ cd /data0/congcong/code/Project_Doing/foundation_model-levelr
$ D=/data2/congcong/data/FM_data/derived/level_r
$ PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/level_r_export.py --out $D 2>&1 | tail -3
164/165 f64e9f4a: 10 lesions, shape (16, 320, 320)
165/165 c07c82a2: 1 lesions, shape (16, 320, 320)
exported 1297 lesions from 165 volumes to /data2/congcong/data/FM_data/derived/level_r
```

```
$ ls $D/volumes | wc -l
330
$ du -sh $D
473M	/data2/congcong/data/FM_data/derived/level_r
```

```
$ python -c "import json; r=json.load(open('$D/lesions.json')); print(len(r), len({x['volume_code'] for x in r}))"
1297 165
```

与计划期望一致：`exported 1297 lesions from 165 volumes`；`volumes/` 330 个文件（165 卷 × 2：`.json` + `.u16`）；`1297 165`。
`match_registry` 未抛错。

## 1. 建库

```
$ python scripts/level_r_admin.py init --db $S/level_r.sqlite --lesions $D/lesions.json
/data2/congcong/data/FM_data/derived/level_r_smoke/level_r.sqlite: 1297 lesions loaded from /data2/congcong/data/FM_data/derived/level_r/lesions.json
```

## 2. 三个冒烟账号（token 已打码）

```
$ python scripts/level_r_admin.py add-reader --db $S/level_r.sqlite --reader-id smoke_r1 --role reader --display "冒烟读者 1" | tee $S/tokens.txt
reader smoke_r1 (reader, 冒烟读者 1) created. Link (shown once, not stored):
  /?token=<token>

$ python scripts/level_r_admin.py add-reader --db $S/level_r.sqlite --reader-id smoke_r2 --role reader --display "冒烟读者 2" | tee -a $S/tokens.txt
reader smoke_r2 (reader, 冒烟读者 2) created. Link (shown once, not stored):
  /?token=<token>

$ python scripts/level_r_admin.py add-reader --db $S/level_r.sqlite --reader-id smoke_adj --role adjudicator --display "冒烟裁定" | tee -a $S/tokens.txt
reader smoke_adj (adjudicator, 冒烟裁定) created. Link (shown once, not stored):
  /?token=<token>
```

真实 token 只在 `$S/tokens.txt`（服务器本地，不入库）。

## 3. 分发顺序（pilot 优先）

```
$ python scripts/level_r_admin.py order --db $S/level_r.sqlite --reader-id smoke_r1 --seed 1 --pilot data/level_r/pilot_150.json
smoke_r1: 1297 lesions, 150 pilot first, seed 1

$ python scripts/level_r_admin.py order --db $S/level_r.sqlite --reader-id smoke_r2 --seed 2 --pilot data/level_r/pilot_150.json
smoke_r2: 1297 lesions, 150 pilot first, seed 2
```

## 4. 起服务（8791，独立冒烟库）

```
$ setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/level_r_server.py \
    --db $S/level_r.sqlite --data-root $D --port 8791 --pid-file $S/server.pid > $S/server.log 2>&1 &
```

`server.log` 首行（确认启动）：

```
Level R serving /data2/congcong/data/FM_data/derived/level_r on http://127.0.0.1:8791/ (db /data2/congcong/data/FM_data/derived/level_r_smoke/level_r.sqlite)
```

PID（`$S/server.pid`）：`1078860`。停止：`kill $(cat /data2/congcong/data/FM_data/derived/level_r_smoke/server.pid)`。
本次冒烟结束后**未停止**，留给用户浏览器验收，验收完由用户自己 kill。

## 5. curl 检查

`T` = smoke_r1 的 token（打码，真实值见 `$S/tokens.txt` 第一行）。

```
$ curl -s -o /dev/null -w "%{http_code} %{content_type}\n" http://127.0.0.1:8791/
200 text/html
```

```
$ curl -s http://127.0.0.1:8791/api/enums | head -c 200; echo
{"primary_hosts": ["white_matter", "cortex", "thalamus", "basal_ganglia", "brainstem", "cerebellum", "other"], "topography": ["periventricular", "juxtacortical", "cortical", "deep_white_matter", "infr
```

```
$ curl -s "http://127.0.0.1:8791/api/me?token=<token>"; echo
{"reader_id": "smoke_r1", "role": "reader", "display": "冒烟读者 1", "done": 0, "total": 1297, "next": 830}
```

`L` = 830（上面 `next` 字段，取代计划里 `curl ... | python -c "...['next']"`，见开头方法学说明第 2 条）。

```
$ curl -s "http://127.0.0.1:8791/api/lesion/830?token=<token>" | head -c 300; echo
{"lesion": {"lesion_id": 830, "code": "c52d695d", "volume_code": "b3e1f6f0", "z0": 7, "z1": 7, "boxes": {"7": [[103, 108, 187, 193]]}}, "answer": null}
```

`lesion` 对象正好 6 个键（`lesion_id`、`code`、`volume_code`、`z0`、`z1`、`boxes`），符合预期。
`V` = `b3e1f6f0`（上面 `volume_code` 字段，同样取代 `curl ... | python -c "...['volume_code']"`）。

```
$ curl -s "http://127.0.0.1:8791/api/volume/b3e1f6f0.json?token=<token>"; echo
{"shape": [16, 320, 320], "spacing_slice_mm": 5.0, "spacing_row_mm": 0.6875, "spacing_col_mm": 0.6875, "window": [3422, 56592]}
```

```
$ curl -s -o /dev/null -w "u16 bytes %{size_download}\n" "http://127.0.0.1:8791/api/volume/b3e1f6f0.u16?token=<token>"
u16 bytes 3276800
```

校验：`16 × 320 × 320 × 2 = 3276800`，与 shape 一致。

```
$ curl -s -o /dev/null -w "bad token -> %{http_code}\n" "http://127.0.0.1:8791/api/me?token=0000000000000000"
bad token -> 403
```

全部与计划期望一致：`200 text/html`；enums JSON；`/api/me` 给出 `total 1297, done 0, next <id>`；
病灶 JSON（`lesion` 子对象）6 个键；卷 JSON 形如 `shape [16, 320, 320]`；u16 字节数 = shape 乘积 × 2；坏 token → 403。

## 6. 医生访问方式（供用户）

端口转发：

```
ssh -L 8791:127.0.0.1:8791 <server>
```

浏览器地址形如：

```
http://127.0.0.1:8791/?token=<token>
```

三个冒烟账号的真实 token 在服务器上的 `/data2/congcong/data/FM_data/derived/level_r_smoke/tokens.txt`（不在仓库里）。

## 7. 浏览器验收（用户做，本任务只登记表格）

| # | 步骤 | 结果 |
|---|------|------|
| 1 | 打开第一例 | USER_REPORTED |
| 2 | 翻层 | USER_REPORTED |
| 3 | 拖窗 | USER_REPORTED |
| 4 | 放大 | USER_REPORTED |
| 5 | 提交一次 | USER_REPORTED |
| 6 | 回列表改一次 | USER_REPORTED |
| 7 | 用冒烟读者 2 再提交一例让它与读者 1 分歧 | USER_REPORTED |
| 8 | 用裁定 token 看到分歧并裁定 | USER_REPORTED |
| 9 | 勾选"不是病灶"后病灶类型/侧别/脑叶三个下拉变灰 | USER_REPORTED |
| 10 | 两位读者只在病灶类型上不一致时该病灶出现在裁定列表 | USER_REPORTED |

这一步不阻塞后续任务；服务不停，等用户看完再 `kill $(cat /data2/congcong/data/FM_data/derived/level_r_smoke/server.pid)`。
