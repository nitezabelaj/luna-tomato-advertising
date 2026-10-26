import numpy as np
from esa_code.lta_udp import udp

T = udp.t
target = udp.target_signal > 0.5
n = len(T)
dt = T[1] - T[0]

print("Pika gjithsej:", n)
print("Pika me vlere 1:", int(target.sum()), "| me vlere 0:", int((~target).sum()))
print("Gabime te lejuara per MSE<=0.05:", int(0.05 * n))
print("Gjeresia e nje dritareje (pika):", round(udp.width / dt, 1))
print("Kohezgjatja e sinjalit (dite):", round(T[-1], 2))
print()


def periudha_ditore(f):
    return (1.0 / f) * udp.TU / 86400.0


familjet = {
    "DRO": udp.frequencies_dro,
    "Lyapunov, dritarja 1": udp.frequencies_lyap[:udp.num_lyap],
    "Lyapunov, dritarja 2": udp.frequencies_lyap[udp.num_lyap:],
    "Axial, dritarja 1": udp.frequencies_axial[:udp.num_axial],
    "Axial, dritarja 2": udp.frequencies_axial[udp.num_axial:],
}

for emri, f in familjet.items():
    p = periudha_ditore(f)
    dritare = T[-1] / p
    print(f"{emri}: {len(f)} orbita")
    print(f"   periudha (dite): min {p.min():.2f} | mesatare {np.median(p):.2f} | max {p.max():.2f}")
    print(f"   dritare ne mesazh: min {dritare.min():.2f} | mesatare {np.median(dritare):.2f} | max {dritare.max():.2f}")