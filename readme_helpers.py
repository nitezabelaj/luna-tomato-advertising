import numpy as np
from esa_code.lta_udp import udp

# Target signal helpers
message = " Tomatoes for sale "
t, target_signal = udp.morse_to_signal(message)
morse_string = udp.message_to_morse_string(message)
print(morse_string)
for emri, db in (("DRO", udp.db_dro), ("Lyapunov", udp.db_lyap), ("Axial", udp.db_axial)):
    period_rad = db[0, 6]
    period_days = period_rad / (2.0 * np.pi) * udp.synodic_period
    print(emri, "| period_rad:", period_rad, "| period_days:", period_days)