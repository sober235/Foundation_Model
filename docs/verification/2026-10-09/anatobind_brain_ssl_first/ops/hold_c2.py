"""Hold 4 GiB on every visible card (2026-10-10: cards 0 and 2) from now, through the end of round 2's Stage I, its G1
on card 5, and C2's smoke with its stop/resume gap, until C2's main Stage II has logged a step (then release after
30 s). Release at once if round 2's chain or C2's chain reports a failure; give up after TIMEOUT seconds."""
import glob, re, sys, time
import torch

GIB, TIMEOUT = 4.0, 8 * 3600
RUNS = "/data2/congcong/data/FM_data/derived/aur/ssl_runs"
R2LOG = f"{RUNS}/stage1_r2_chain_20261010_1303.log"
FAIL_R2 = re.compile(r"did not pass|script failed|ended without|stopped early|chain done")
FAIL_C2 = re.compile(r"failed|does not split|another c2 chain|ended at step")


def log(m):
    print(time.strftime("[%F %T]"), m, flush=True)


def text(path):
    try:
        return open(path).read()
    except OSError:
        return ""


held = []
for d in range(torch.cuda.device_count()):
    x = torch.empty(int(GIB * 2 ** 30), dtype=torch.uint8, device=f"cuda:{d}")
    x.fill_(0)
    held.append(x)
torch.cuda.synchronize()
log(f"holding {GIB} GiB on each of {torch.cuda.device_count()} visible cards")
t0 = time.time()
while time.time() - t0 < TIMEOUT:
    if FAIL_R2.search(text(R2LOG)):
        log("round 2's chain reports an end or a failure: releasing now"); break
    c2logs = sorted(glob.glob(f"{RUNS}/c2_chain_*.log"))
    if c2logs and FAIL_C2.search(text(c2logs[-1])):
        log(f"C2's chain reports a failure ({c2logs[-1]}): releasing now"); break
    stage2 = [p for p in glob.glob(f"{RUNS}/c2_stage2_240k_*/log_rank0.jsonl") if text(p).strip()]
    if stage2:
        log(f"C2's Stage II has logged a step ({stage2[0]}): releasing in 30 s"); time.sleep(30); break
    time.sleep(15)
else:
    log("timeout: releasing")
log("released")
