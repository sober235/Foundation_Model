"""Hold GIB on each visible card until every listed physical card (nvidia-smi index) uses more than NEED_MIB, i.e. the
restarted job has allocated on all of them; then release after 30 s. Gives up after TIMEOUT s.
Usage: hold_multi.py GIB NEED_MIB TIMEOUT CARD [CARD ...]"""
import subprocess, sys, time
import torch
gib, need, timeout, cards = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
def log(m): print(time.strftime("[%F %T]"), m, flush=True)
def used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    d = dict(l.split(", ") for l in out.strip().splitlines())
    return {c: int(d[c]) for c in cards}
held = [torch.empty(int(gib * 2 ** 30), dtype=torch.uint8, device=f"cuda:{i}").fill_(0) for i in range(torch.cuda.device_count())]
torch.cuda.synchronize()
log(f"holding {gib} GiB on cards {cards}: {used()}")
t0 = time.time()
while time.time() - t0 < timeout:
    u = used()
    if all(v > need for v in u.values()):
        log(f"all cards above {need} MiB ({u}): releasing in 30 s"); time.sleep(30); break
    time.sleep(5)
else:
    log("timeout: releasing")
log("released")
