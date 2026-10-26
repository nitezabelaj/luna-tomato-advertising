import numpy as np
from esa_code.lta_udp import udp

# Target signal helpers
message = " Tomatoes for sale "
t, target_signal = udp.morse_to_signal(message)
morse_string = udp.message_to_morse_string(message)
print(morse_string)