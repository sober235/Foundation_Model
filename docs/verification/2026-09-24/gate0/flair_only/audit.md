# Gate 0 box audit

| organ | volumes | converted wins | as-is wins | median ratio as-is | median ratio converted |
|---|---|---|---|---|---|
| knee | 236 | 211 | 25 | 1.292 | 1.661 |
| brain | 165 | 158 | 7 | 1.051 | 1.147 |

brain per series (n, converted wins, RSS rows):

- 200: n=59, converted=58, rows=[320]
- 201: n=41, converted=38, rows=[320]
- 202: n=17, converted=17, rows=[256, 320]
- 203: n=19, converted=17, rows=[213, 234, 276]
- 205: n=1, converted=1, rows=[320]
- 206: n=4, converted=4, rows=[256]
- 209: n=11, converted=10, rows=[320]
- 210: n=13, converted=13, rows=[320]

GATE0_AUDIT: FAIL (knee >= 90%, brain >= 90%, every series with >= 3 volumes >= 75%)
