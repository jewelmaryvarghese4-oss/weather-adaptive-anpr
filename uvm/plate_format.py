"""
Post-processes a raw LPRNet prediction using the known structure of an
Indian license plate: state code (2 letters) + RTO code (2 digits) +
series (1 or 2 letters) + number (4 digits) -- e.g. KL 01 BN 0345.

Why this helps: CTC's greedy decode picks the single highest-probability
class at each position independently, with no notion of "this position
must be a letter" or "must be a digit". That's exactly where visually
confusable pairs slip through -- O/0, I/1, S/5, Z/2, G/6, B/8 look alike
enough that the network's raw argmax gets the SHAPE right but the
CLASS (letter vs digit) wrong. Since the plate format fixes which class
every position must be, any position whose predicted character has the
wrong class for its slot is very likely one of these confusable-pair
mistakes, not a genuinely different character -- so it can be corrected
by swapping to the confusable counterpart in the right class.

This only fires when the prediction's length exactly matches a known
Indian plate format (9 or 10 characters) -- it never reshuffles or
guesses on a prediction that doesn't already look like a plate, since
forcing a format onto noise would just invent wrong text with false
confidence.

apply_format_constraint() (string-only) uses a FIXED swap table, so it
only ever offers one replacement candidate per wrong-class character
(e.g. digit '0' always becomes letter 'O'). That's too rigid -- a real
case showed the network reading a 'Q' as digit '0' get "corrected" to
'O' instead, because 'Q' was never in the table as a candidate at all.
ctc_format_constrained_decode() fixes this properly: instead of a fixed
table, it goes back to the network's own per-position probabilities and
picks whichever LETTER (or digit) it actually favored most at that
position -- so it can recover 'Q', or anything else, not just the one
hardcoded lookalike. Prefer that version whenever log_probs are
available (i.e. whenever you're calling the model yourself, not just
post-processing an already-decoded string from elsewhere).

Neither version can fix a same-class confusion (e.g. 'V' misread as
'Y' -- both letters) since the plate format only constrains which CLASS
a position must be, not which specific character within that class.
That kind of mistake means the base model itself needs to get better at
telling those two shapes apart -- more/better training data, not a
decode-time constraint.
"""

# state(2 letters) + RTO(2 digits) + series(2 letters) + number(4 digits)
FORMAT_10 = "LLDDLLDDDD"
# state(2 letters) + RTO(2 digits) + series(1 letter) + number(4 digits)
FORMAT_9 = "LLDDLDDDD"
# state(2 letters) + RTO(1 digit) + series(2 letters) + number(4 digits) -- Delhi style, DL 1Z C 8889
FORMAT_9_ONE_DIGIT_RTO = "LLDLLDDDD"

# Several layouts can share a length (9: KA08A0339 vs DL1ZC8889), so each length maps to a
# list; the decoders below pick whichever layout needs the fewest/cheapest corrections
# (zero corrections = the reading already fits, so it is returned unchanged).
# state(2 letters) + RTO(2 digits) + series(2 letters) + number(3 digits) -- older / low-number plates,
# e.g. 'KL 01 / CJ.145'. Without it the 9-character LLDDLDDDD layout "fixes" the second series letter
# into a digit (CJ145 -> C1145).
FORMAT_9_THREE_DIGIT_NUMBER = "LLDDLLDDD"

# state(2 letters) + RTO(1 digit + 1 letter) + series(2 letters) + number(4 digits) -- Delhi style with a lettered RTO
# code, e.g. 'DL 3C CU 5010', 'DL 9C AQ 9338'. Without it the 10-character LLDDLLDDDD layout forced the 'C' of
# '3C' into a digit (DL3CCU5010 -> DL36CU5010).
FORMAT_10_DELHI_LETTER_RTO = "LLDLLLDDDD"

FORMATS_BY_LENGTH = {10: [FORMAT_10, FORMAT_10_DELHI_LETTER_RTO],
                     9: [FORMAT_9, FORMAT_9_ONE_DIGIT_RTO, FORMAT_9_THREE_DIGIT_NUMBER]}

# Visually confusable pairs a CTC argmax is prone to swap between,
# keyed by the WRONG-class character -> its same-shape counterpart in
# the class the plate format actually calls for at that position.
_LETTER_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "Q": "0"}
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}


def apply_format_constraint(text: str) -> str:
    """Returns a corrected copy of text if its length matches a known
    Indian plate format and any position's character has the wrong
    class (letter where a digit belongs, or vice versa) with a known
    confusable counterpart. Otherwise returns text unchanged."""
    fmts = FORMATS_BY_LENGTH.get(len(text))
    if fmts is None:
        return text

    def wrong_class(fmt):
        return sum((e == "D") != c.isdigit() for c, e in zip(text, fmt))

    fmt = min(fmts, key=wrong_class)  # first one wins ties

    corrected = list(text)
    for i, (ch, expected) in enumerate(zip(text, fmt)):
        is_digit = ch.isdigit()
        if expected == "D" and not is_digit and ch in _LETTER_TO_DIGIT:
            corrected[i] = _LETTER_TO_DIGIT[ch]
        elif expected == "L" and is_digit and ch in _DIGIT_TO_LETTER:
            corrected[i] = _DIGIT_TO_LETTER[ch]
    return "".join(corrected)


def ctc_format_constrained_decode(log_probs) -> str:
    """log_probs: (T, 1, C) tensor for a SINGLE image (batch size 1, as
    read_plate.py processes one plate crop at a time) -- the same log-
    softmax output CTCLoss/ctc_greedy_decode expect.

    Does the same greedy CTC decode as lprnet_vocab.ctc_greedy_decode
    (argmax per timestep, collapse repeats, drop blanks), but additionally
    remembers WHICH timestep produced each output character. If the
    resulting text's length matches a known Indian plate format, each
    character is then re-derived from that same timestep's real
    probabilities, restricted to only the classes valid for that
    position (letters-only or digits-only) -- so it reflects what the
    network actually favored among the *right* class, rather than a
    fixed lookup-table swap. See this module's docstring for why this
    matters (recovers cases like 'Q' that a fixed table doesn't cover)
    and what it still can't fix (same-class confusions like V/Y)."""
    from lprnet_vocab import BLANK, CHAR_TO_IDX, IDX_TO_CHAR

    pred = log_probs[:, 0, :].argmax(dim=1).tolist()
    chars, timesteps = [], []
    prev = None
    for t, idx in enumerate(pred):
        if idx != prev and idx != BLANK:
            chars.append(IDX_TO_CHAR[idx])
            timesteps.append(t)
        prev = idx

    fmts = FORMATS_BY_LENGTH.get(len(chars))
    if fmts is None:
        return "".join(chars)

    letter_idx = list(range(1, 27))  # CHAR_TO_IDX['A'..'Z'] -- see lprnet_vocab.py
    digit_idx = list(range(27, 37))  # CHAR_TO_IDX['0'..'9']

    def cost(fmt):
        """How much log-probability the network loses if forced into this layout
        (0 when its own reading already fits)."""
        total = 0.0
        for expected, t, idx in zip(fmt, timesteps, pred_idx):
            allowed = letter_idx if expected == "L" else digit_idx
            total += log_probs[t, 0, idx].item() - log_probs[t, 0, allowed].max().item()
        return total  # each term >= 0: the greedy pick is the global max at its timestep

    pred_idx = [CHAR_TO_IDX[c] for c in chars]
    fmt = min(fmts, key=cost)  # first one wins ties

    corrected = list(chars)
    for i, (expected, t) in enumerate(zip(fmt, timesteps)):
        allowed = letter_idx if expected == "L" else digit_idx
        best = allowed[log_probs[t, 0, allowed].argmax().item()]
        corrected[i] = IDX_TO_CHAR[best]
    return "".join(corrected)
