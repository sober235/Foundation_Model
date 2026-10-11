"""Hold GIB on every visible card across a restart (2026-10-11): wait until all OLD_PIDS have exited, then until every
listed physical card (nvidia-smi index) uses more than NEED_MIB (the new job has allocated), release after 30 s; give up
after TIMEOUT s. Usage: hold_swap.py GIB NEED_MIB TIMEOUT OLD_PID[,OLD_PID...] CARD [CARD ...]"""
import os, subprocess, sys, time
import torch
gib, need, timeout, olds, cards = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), [int(p) for p in sys.argv[4].split(",")], sys.argv[5:]
def log(m): print(time.strftime("[%F %T]"), m, flush=True)
def used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    d = dict(l.split(", ") for l in out.strip().splitlines())
    return {c: int(d[c]) for c in cards}
held = [torch.empty(int(gib * 2 ** 30), dtype=torch.uint8, device=f"cuda:{i}").fill_(0) for i in range(torch.cuda.device_count())]
torch.cuda.synchronize()
log(f"holding {gib} GiB on cards {cards} ({used()}); waiting for pids {olds} to exit")
t0 = time.time()
while any(os.path.exists(f"/proc/{p}") for p in olds) and time.time() - t0 < timeout:
    time.sleep(2)
log(f"old pids gone; cards {used()}; waiting for every card above {need} MiB")
while time.time() - t0 < timeout:
    u = used()
    if all(v > need for v in u.values()):
        log(f"all cards above {need} MiB ({u}): releasing in 30 s"); time.sleep(30); break
    time.sleep(5)
else:
    log("timeout: releasing")
log("released")
