# Gate 0 box audit

| organ | volumes | converted wins | as-is wins | median ratio as-is | median ratio converted |
|---|---|---|---|---|---|
| knee | 236 | 211 | 25 | 1.292 | 1.661 |
| brain | 188 | 166 | 22 | 1.052 | 1.144 |

brain per series (n, converted wins, RSS rows):

- 200: n=63, converted=59, rows=[320]
- 201: n=56, converted=43, rows=[320]
- 202: n=21, converted=19, rows=[256, 320]
- 203: n=19, converted=17, rows=[213, 234, 276]
- 205: n=1, converted=1, rows=[320]
- 206: n=4, converted=4, rows=[256]
- 209: n=11, converted=10, rows=[320]
- 210: n=13, converted=13, rows=[320]

GATE0_AUDIT: FAIL (knee >= 90%, brain >= 90%, every series with >= 3 volumes >= 75%)
