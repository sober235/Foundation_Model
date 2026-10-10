"""Card guard (2026-10-10): while a chain is alive (the pid in LOCKFILE), keep each listed card from looking free in the
gaps between its jobs (Stage II -> Stage III -> evaluation shards -> merge -> test shards). Every 10 s: if the memory
the chain's own jobs use on a card (total minus this guard's share) is under 20 GB, hold 4 GiB there; once it has been
over 20 GB for 30 s, release the hold. Exits (releasing everything) when the chain has been gone for 30 s.
Usage: CUDA_VISIBLE_DEVICES=<cards> card_guard.py LOCKFILE CARD [CARD ...]   (cards in the same order)"""
import os, subprocess, sys, time
import torch

GIB, OTHER_MIB, CTX_MIB = 4.0, 20000, 600
lock, cards = sys.argv[1], sys.argv[2:]


def log(m):
    print(time.strftime("[%F %T]"), m, flush=True)


def alive():
    try:
        os.kill(int(open(lock).read().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    d = dict(l.split(", ") for l in out.strip().splitlines())
    return [int(d[c]) for c in cards]


held = [None] * len(cards)
ctx = [False] * len(cards)
busy_for = [0] * len(cards)
dead_for = 0
log(f"guarding cards {cards} for the chain in {lock}")
while True:
    if not alive():
        dead_for += 1
        if dead_for >= 3:
            log("the chain is gone: releasing and exiting")
            break
    else:
        dead_for = 0
    u = used()
    for i, c in enumerate(cards):
        mine = (int(GIB * 1024) if held[i] is not None else 0) + (CTX_MIB if ctx[i] else 0)
        other = u[i] - mine
        if other < OTHER_MIB:
            busy_for[i] = 0
            if held[i] is None:
                held[i] = torch.empty(int(GIB * 2 ** 30), dtype=torch.uint8, device=f"cuda:{i}").fill_(0)
                torch.cuda.synchronize(i)
                ctx[i] = True
                log(f"card {c}: the chain's jobs use {other} MiB: holding {GIB} GiB")
        else:
            busy_for[i] += 1
            if held[i] is not None and busy_for[i] >= 3:
                held[i] = None
                torch.cuda.empty_cache()
                log(f"card {c}: the chain's jobs use {other} MiB: hold released")
    time.sleep(10)
held = [None] * len(cards)
torch.cuda.empty_cache()
log("exited")
