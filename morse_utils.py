
MORSE_CODE = {
    "A": ".-",    "B": "-...",  "C": "-.-.",
    "D": "-..",   "E": ".",     "F": "..-.",
    "G": "--.",   "H": "....",  "I": "..",
    "J": ".---",  "K": "-.-",   "L": ".-..",
    "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",
    "S": "...",   "T": "-",     "U": "..-",
    "V": "...-",  "W": ".--",   "X": "-..-",
    "Y": "-.--",  "Z": "--..",

    "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....",
    "6": "-....", "7": "--...", "8": "---..",
    "9": "----.",

    " ": "/"
}


def text_to_morse(message):
    message = message.upper()
    encoded_characters = []

    for character in message:
        if character not in MORSE_CODE:
            raise ValueError(
                f"Character not supported: {character!r}"
            )

        encoded_characters.append(MORSE_CODE[character])

    return " ".join(encoded_characters)



def morse_to_signal(morse_message, samples_per_unit=10):
    """Convert Morse code into a binary time signal."""

    if samples_per_unit < 1:
        raise ValueError("samples_per_unit must be at least 1")

    signal = []

    dot_duration = 1
    dash_duration = 3
    symbol_gap = 1
    letter_gap = 3
    word_gap = 7

    words = morse_message.strip().split(" / ")

    for word_index, word in enumerate(words):
        letters = word.split()

        for letter_index, letter in enumerate(letters):
            for symbol_index, symbol in enumerate(letter):

                if symbol == ".":
                    duration = dot_duration
                elif symbol == "-":
                    duration = dash_duration
                else:
                    raise ValueError(
                        f"Invalid Morse symbol: {symbol}"
                    )

                signal.extend(
                    [1] * (duration * samples_per_unit)
                )

                if symbol_index < len(letter) - 1:
                    signal.extend(
                        [0] * (symbol_gap * samples_per_unit)
                    )

            if letter_index < len(letters) - 1:
                signal.extend(
                    [0] * (letter_gap * samples_per_unit)
                )

        if word_index < len(words) - 1:
            signal.extend(
                [0] * (word_gap * samples_per_unit)
            )

    return signal
