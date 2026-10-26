import numpy as np
from esa_code.lta_udp import udp


udp.plot_target()


lb, ub = udp.get_bounds()
rng = np.random.default_rng(42)
x = rng.uniform(lb, ub)


obj, num_selected, mse = udp.fitness(x, postprocess=True)
print("Objektivi:", obj)
print("Numri i anijeve:", num_selected)
print("MSE:", mse, "| kufiri i lejuar:", udp.mse_thresh)


udp.plot_signal(x)
import time
t0 = time.time()
udp.fitness(x)
print("Koha për një fitness:", round(time.time() - t0, 3), "sekonda")