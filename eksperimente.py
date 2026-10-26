import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from esa_code.lta_udp import udp

plt.switch_backend("Agg")

T = udp.t
dt = T[1] - T[0]
target = udp.target_signal > 0.5
NT = int(target.sum())
NTOT = len(T)
M = udp.max_per_orbit
ND, NL, NA = udp.num_dro, udp.num_lyap, udp.num_axial
N = udp.num_orbits
TWO_PI = 2 * np.pi
WIN = int(round(udp.width / dt))
CACHE = "kandidatet_fp0.npz"
EMRAT = {0: "DRO", 1: "Lyapunov", 2: "Axial"}
pulses, i = [], 0
while i < len(target):
    if target[i]:
        j = i
        while j < len(target) and target[j]:
            j += 1
        pulses.append((i, j))
        i = j
    else:
        i += 1
tiles = [s for a, b in pulses for s in range(a, b, WIN)]
c_nd = np.array([(s + (WIN - 1) / 2) * dt * 24 * 3600 / udp.TU for s in tiles])
def win(centers, f):
    return udp._repeating_box_window(T[None, :], centers[:, None], f) > 0.5


def maske(p, f1, f2=None):
    m = win((TWO_PI - p) / TWO_PI / f1, f1)
    if f2 is not None:
        m = m | win((p + np.pi) / f2, f2)
    return m


if os.path.exists(CACHE):
    d = np.load(CACHE)
    orb, fam, ph, P = d["orb"], d["fam"], d["ph"], d["P"]
    print("Kandidatet u ngarkuan nga", CACHE)
else:
    t0 = time.time()
    L_orb, L_fam, L_ph, L_P = [], [], [], []

    def shto(g, fm, p, m):
        fp = (m & ~target[None, :]).sum(axis=1)
        cov = (m & target[None, :]).sum(axis=1)
        keep = (fp == 0) & (cov >= 15)
        if keep.any():
            L_orb.append(np.full(int(keep.sum()), g))
            L_fam.append(np.full(int(keep.sum()), fm))
            L_ph.append(p[keep])
            L_P.append(np.packbits(m[keep], axis=1))

    for i in range(ND):
        f = udp.frequencies_dro[i]
        p = (TWO_PI * (1 - c_nd * f)) % TWO_PI
        shto(i, 0, p, maske(p, f))
    for fm, n, off, fr in ((1, NL, ND, udp.frequencies_lyap),
                           (2, NA, ND + NL, udp.frequencies_axial)):
        for i in range(n):
            f1, f2 = fr[i], fr[n + i]
            pA = (TWO_PI * (1 - c_nd * f1)) % TWO_PI
            base = c_nd * f2 - np.pi
            pB = ((base % 1.0)[:, None] + np.arange(7)[None, :]).ravel()
            pB = pB[pB < TWO_PI]
            p = np.concatenate([pA, pB])
            shto(off + i, fm, p, maske(p, f1, f2))
        print("Mbaroi", EMRAT[fm])
    orb, fam = np.concatenate(L_orb), np.concatenate(L_fam)
    ph, P = np.concatenate(L_ph), np.vstack(L_P)
    np.savez_compressed(CACHE, orb=orb, fam=fam, ph=ph, P=P)
    print(f"Kandidatet u gjeneruan per {time.time() - t0:.0f} s dhe u ruajten")

for fm in (0, 1, 2):
    print("Kandidate", EMRAT[fm], ":", int((fam == fm).sum()))

#--- Algoritmet----
POP = np.array([bin(k).count("1") for k in range(256)], dtype=np.uint8)


def pop(a):
    return int(POP[a].sum(dtype=np.int64))


def lakmitare(o, Q, stop_err):
    """Zgjedh kandidatin qe mbulon me shume pika te reja, derisa gabimet <= stop_err."""
    covered = np.zeros(Q.shape[1], np.uint8)
    cnt = np.zeros(N, int)
    chosen = []
    while NT - pop(covered) > stop_err:
        gain = POP[Q & ~covered[None, :]].sum(axis=1, dtype=np.int32)
        gain[cnt[o] >= M] = 0
        k = int(np.argmax(gain))
        if gain[k] == 0:
            break
        chosen.append(k)
        covered |= Q[k]
        cnt[o[k]] += 1
    return chosen


