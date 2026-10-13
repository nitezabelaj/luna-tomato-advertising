

import matplotlib.pyplot as plt

from morse_utils import text_to_morse, morse_to_signal


def main():
    message = "SOS"

    morse_message = text_to_morse(message)
    signal = morse_to_signal(morse_message)

    print("Mesazhi:", message)
    print("Kodi Morse:", morse_message)
    print("Numri i mostrave:", len(signal))
    print("Vlerat e sinjalit:", set(signal))

    time = list(range(len(signal)))

    plt.figure(figsize=(12, 4))
    plt.step(time, signal, where="post")
    plt.title(f"Sinjali Morse për mesazhin {message}")
    plt.xlabel("Indeksi i mostrës")
    plt.ylabel("Gjendja e sinjalit")
    plt.yticks([0, 1], ["Joaktiv (0)", "Aktiv (1)"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
