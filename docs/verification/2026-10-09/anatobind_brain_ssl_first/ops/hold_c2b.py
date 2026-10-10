"""Hold 4 GiB on every visible card (cards 0 and 2) for C2 through the end of round 2's training, its G1 on card 5 and
C2's smoke with its stop/resume gap; release 30 s after C2's main Stage II has logged a step, at once if the newest
round-2 chain log or C2's chain log reports a failure; give up after 8 h (2026-10-10)."""
import glob, os, re, time
import torch

GIB, TIMEOUT = 4.0, 8 * 3600
RUNS = "/data2/congcong/data/FM_data/derived/aur/ssl_runs"
FAIL_R2 = re.compile(r"did not pass|script failed|ended without")
FAIL_C2 = re.compile(r"failed|does not split|another c2 chain|ended at step")


def log(m):
    print(time.strftime("[%F %T]"), m, flush=True)


def newest(pattern):
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    return paths[-1] if paths else None


def text(path):
    try:
        return open(path).read() if path else ""
    except OSError:
        return ""


held = [torch.empty(int(GIB * 2 ** 30), dtype=torch.uint8, device=f"cuda:{d}").fill_(0) for d in range(torch.cuda.device_count())]
torch.cuda.synchronize()
log(f"holding {GIB} GiB on each of {len(held)} visible cards for C2")
t0 = time.time()
while time.time() - t0 < TIMEOUT:
    r2 = newest(f"{RUNS}/stage1_r2_resume_chain_*.log")
    if FAIL_R2.search(text(r2)):
        log(f"{r2} reports an end or a failure: releasing now"); break
    c2 = newest(f"{RUNS}/c2_chain_*.log")
    if FAIL_C2.search(text(c2)):
        log(f"{c2} reports a failure: releasing now"); break
    logged = [p for p in glob.glob(f"{RUNS}/c2_stage2_240k_*/log_rank0.jsonl") if text(p).strip()]
    if logged:
        log(f"C2's Stage II has logged a step ({logged[0]}): releasing in 30 s"); time.sleep(30); break
    time.sleep(15)
else:
    log("timeout: releasing")
log("released")
