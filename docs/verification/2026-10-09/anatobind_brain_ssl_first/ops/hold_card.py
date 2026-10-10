"""Hold GIB of the visible card across a restart (2026-10-10): wait until OLD_PID has exited, then until the card's
used memory (nvidia-smi index CARD) exceeds NEED_MIB, i.e. the restarted job has allocated, then release after 30 s.
Gives up after TIMEOUT seconds. Usage: hold_card.py GIB OLD_PID CARD NEED_MIB TIMEOUT"""
import os, subprocess, sys, time
import torch

gib, old_pid, card, need_mib, timeout = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5])


def log(m):
    print(time.strftime("[%F %T]"), m, flush=True)


def used():
    out = subprocess.run(["nvidia-smi", "-i", card, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip())


x = torch.empty(int(gib * 2 ** 30), dtype=torch.uint8, device="cuda")
x.fill_(0)
torch.cuda.synchronize()
t0 = time.time()
log(f"holding {gib} GiB on card {card} (card at {used()} MiB); waiting for pid {old_pid} to exit")
while os.path.exists(f"/proc/{old_pid}") and time.time() - t0 < timeout:
    time.sleep(2)
log(f"pid {old_pid} gone; card at {used()} MiB; waiting for more than {need_mib} MiB")
while used() <= need_mib and time.time() - t0 < timeout:
    time.sleep(5)
log(f"card at {used()} MiB after {time.time() - t0:.0f} s; releasing in 30 s")
time.sleep(30)
log("released")
