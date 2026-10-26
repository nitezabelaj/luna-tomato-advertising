import os
import numpy as np
import matplotlib.pyplot as plt
from esa_code.lta_udp import udp

plt.switch_backend("Agg")  # ruan imazhe, nuk hap dritare

T = udp.t
dt = T[1] - T[0]
target = udp.target_signal > 0.5
M = udp.max_per_orbit
N = udp.num_orbits
ND = udp.num_dro
TWO_PI = 2 * np.pi
WIN = int(round(udp.width / dt))  # 10 pika

# 1) Ndaj cdo puls ne pllaka 10-pikeshe
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
print("Pulse:", len(pulses), "| pllaka:", len(tiles), "| gjeresia e dritares:", WIN)


# 2) Dritarja e nje anijeje DRO me qender c (dite)
def dritare_dro(i, c_days):
    f = udp.frequencies_dro[i]
    c_nd = c_days * 24 * 3600 / udp.TU
    p = (TWO_PI * (1 - c_nd * f)) % TWO_PI
    col = udp.amplitudes_dro[i] * udp._repeating_box_window(
        T, (TWO_PI - p) / TWO_PI / f, frequency=f)
    return p, col > 0.5


# 3) Per cdo pllake gjej orbitat DRO qe nuk bien mbi zerot
valid = {}
for k, s in enumerate(tiles):
    c = (s + (WIN - 1) / 2) * dt
    L = []
    for i in range(ND):
        p, m = dritare_dro(i, c)
        if np.sum(m & ~target) == 0 and np.sum(m & target) >= WIN - 1:
            L.append((i, p))
    valid[k] = L
print("Pllaka pa asnje orbite te vlefshme:", sum(1 for v in valid.values() if not v))

# 4) Caktim i lakuar: pllakat me me pak zgjedhje se pari
cap = np.full(ND, M)
sel = np.zeros(N)
phs = np.zeros(N * M)
caktuar = 0
for k in sorted(range(len(tiles)), key=lambda k: len(valid[k])):
    for i, p in sorted(valid[k], key=lambda t: -cap[t[0]]):
        if cap[i] > 0:
            phs[i * M + int(sel[i])] = p
            sel[i] += 1
            cap[i] -= 1
            caktuar += 1
            break
print("Pllaka te mbuluara:", caktuar, "nga", len(tiles))

# 5) Vleresimi zyrtar
x = np.concatenate([sel, phs])
obj, n_ships, mse = udp.fitness(x, postprocess=True)
print("Anije:", n_ships, "| MSE:", mse, "| kufiri:", udp.mse_thresh)
print("E vlefshme:", mse <= udp.mse_thresh)

os.makedirs("solutions", exist_ok=True)
np.save("solutions/baseline_x.npy", x)

# 6) Imazh krahasimi
A = udp._build_occultation_matrix(*udp._decode_chromosome(x))
rec = np.clip(A @ np.ones(A.shape[1]), 0, 1)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
a1.plot(T, udp.target_signal)
a1.set_title("Sinjali i synuar")
a2.plot(T, rec)
a2.set_title(f"I rindertuar | anije {n_ships} | MSE {mse:.4f}")
plt.tight_layout()
plt.savefig("baseline.png", dpi=120)
print("U ruajt baseline.png")