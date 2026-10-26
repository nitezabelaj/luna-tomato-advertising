import os
import numpy as np
import matplotlib.pyplot as plt
from esa_code.lta_udp import udp

plt.switch_backend("Agg")

T = udp.t
dt = T[1] - T[0]
target = udp.target_signal > 0.5
NT = int(target.sum())
M = udp.max_per_orbit
ND, NL, NA = udp.num_dro, udp.num_lyap, udp.num_axial
N = udp.num_orbits
TWO_PI = 2 * np.pi
WIN = int(round(udp.width / dt))
MAX_ERR = 120      # gabime te lejuara ne fund (120/2710 = 0.044)
STOP_ERR = 100     # ndalo kerkimin lakmitar kur mbeten kaq pika pa mbuluar

# --- pllakat 10-pikeshe te pulseve ---
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
print("Pllaka:", len(tiles), "| pika '1':", NT)


def win(centers, f):
    return udp._repeating_box_window(T[None, :], centers[:, None], f) > 0.5


def maske(p, f1, f2=None):
    m = win((TWO_PI - p) / TWO_PI / f1, f1)
    if f2 is not None:
        m = m | win((p + np.pi) / f2, f2)
    return m


# --- gjenero kandidatet ---
orb_list, ph_list, pack_list = [], [], []


def shto(g, p, m):
    fp = (m & ~target[None, :]).sum(axis=1)
    cov = (m & target[None, :]).sum(axis=1)
    keep = (fp == 0) & (cov >= 15)
    if keep.any():
        orb_list.append(np.full(int(keep.sum()), g))
        ph_list.append(p[keep])
        pack_list.append(np.packbits(m[keep], axis=1))


for i in range(ND):
    f = udp.frequencies_dro[i]
    p = (TWO_PI * (1 - c_nd * f)) % TWO_PI
    shto(i, p, maske(p, f))

for fam, n, off, fr in (("lyap", NL, ND, udp.frequencies_lyap),
                        ("axial", NA, ND + NL, udp.frequencies_axial)):
    for i in range(n):
        f1, f2 = fr[i], fr[n + i]
        pA = (TWO_PI * (1 - c_nd * f1)) % TWO_PI
        base = c_nd * f2 - np.pi
        pB = ((base % 1.0)[:, None] + np.arange(7)[None, :]).ravel()
        pB = pB[pB < TWO_PI]
        p = np.concatenate([pA, pB])
        shto(off + i, p, maske(p, f1, f2))
    print("Mbaroi familja:", fam)

orb = np.concatenate(orb_list)
phs_c = np.concatenate(ph_list)
P = np.vstack(pack_list)
print("Kandidate te vlefshem:", len(orb))

# --- zgjedhje lakmitare ---
POP = np.array([bin(k).count("1") for k in range(256)], dtype=np.uint16)
covered = np.zeros(P.shape[1], dtype=np.uint8)
cnt = np.zeros(N, dtype=int)
chosen = []
while True:
    gain = POP[P & ~covered[None, :]].sum(axis=1)
    gain[cnt[orb] >= M] = 0
    k = int(np.argmax(gain))
    if gain[k] == 0:
        break
    chosen.append(k)
    covered |= P[k]
    cnt[orb[k]] += 1
    mbetur = NT - int(POP[covered].sum())
    if len(chosen) % 10 == 0:
        print("Anije:", len(chosen), "| pika te pambuluara:", mbetur)
    if mbetur <= STOP_ERR:
        break
print("Pas zgjedhjes lakmitare:", len(chosen), "anije")

# --- heqje e anijeve te tepert ---
masks = np.unpackbits(P[chosen], axis=1)[:, :len(T)].astype(bool)
counts = masks.sum(axis=0)
active = list(range(len(chosen)))


def gabime(c):
    return int(np.sum(target & (c == 0)))


ndryshoi = True
while ndryshoi:
    ndryshoi = False
    unik = {j: int((masks[j] & (counts == 1)).sum()) for j in active}
    for j in sorted(active, key=lambda j: unik[j]):
        c2 = counts - masks[j]
        if gabime(c2) <= MAX_ERR:
            counts = c2
            active.remove(j)
            ndryshoi = True
            break
print("Pas heqjes se te tepertave:", len(active), "anije")

# --- ndertimi i x dhe vleresimi zyrtar ---
sel = np.zeros(N)
phs = np.zeros(N * M)
for j in active:
    g = orb[chosen[j]]
    phs[g * M + int(sel[g])] = phs_c[chosen[j]]
    sel[g] += 1
x = np.concatenate([sel, phs])

obj, n_ships, mse = udp.fitness(x, postprocess=True)
print("Anije:", n_ships, "| MSE:", mse, "| e vlefshme:", mse <= udp.mse_thresh)
print("DRO:", int(sel[:ND].sum()), "| Lyapunov:", int(sel[ND:ND + NL].sum()),
      "| Axial:", int(sel[ND + NL:].sum()))

os.makedirs("solutions", exist_ok=True)
np.save("solutions/kerkimi_x.npy", x)

A = udp._build_occultation_matrix(*udp._decode_chromosome(x))
rec = np.clip(A @ np.ones(A.shape[1]), 0, 1)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
a1.plot(T, udp.target_signal)
a1.set_title("Sinjali i synuar")
a2.plot(T, rec)
a2.set_title(f"I rindertuar | anije {n_ships} | MSE {mse:.4f}")
plt.tight_layout()
plt.savefig("kerkimi.png", dpi=120)
print("U ruajt kerkimi.png dhe solutions/kerkimi_x.npy")