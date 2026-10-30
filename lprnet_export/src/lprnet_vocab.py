"""
Character vocabulary + CTC encode/decode for LPRNet.

CTC (Connectionist Temporal Classification) needs one extra "blank" class
beyond the real alphabet -- it's how the network says "nothing new here,
still the same character as last column" or "gap between two characters",
which is what lets a fixed-length column sequence (T=23) represent a
variable-length label (1-11 characters) without knowing in advance where
each character starts. Blank is index 0 by convention here.

Alphabet is exactly what's in evaluation/baseline_ocr/ground_truth.json:
A-Z and 0-9, 36 characters, no spaces or punctuation (labels are already
upper-cased and space-stripped -- see label_helper.py).
"""

BLANK = 0
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}  # +1: 0 is reserved for blank
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET) + 1  # 37


def encode(text: str) -> list[int]:
    """Text -> list of class indices (no blanks -- those are what CTC
    loss inserts implicitly during training, not something to encode)."""
    return [CHAR_TO_IDX[c] for c in text]


def ctc_greedy_decode(log_probs) -> list[str]:
    """log_probs: (T, B, num_classes) tensor (the shape CTCLoss expects).
    Returns one decoded string per batch item: take the argmax class at
    each column, then collapse consecutive duplicate classes into one and
    drop blanks -- the standard CTC decoding rule (e.g. "AA-BB-B" with
    '-' as blank collapses to "ABB", not "AABBB" or "AB")."""
    import torch

    pred = log_probs.argmax(dim=2)  # (T, B)
    pred = pred.transpose(0, 1)  # (B, T)
    texts = []
    for seq in pred:
        seq = seq.tolist()
        collapsed = []
        prev = None
        for idx in seq:
            if idx != prev and idx != BLANK:
                collapsed.append(IDX_TO_CHAR[idx])
            prev = idx
        texts.append("".join(collapsed))
    return texts
