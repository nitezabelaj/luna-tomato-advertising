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