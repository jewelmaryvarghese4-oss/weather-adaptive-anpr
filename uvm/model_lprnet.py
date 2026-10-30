"""
LPRNet: a lightweight CNN purpose-built for license plate character
recognition (Zherzdev & Gruzdev, 2018), reading the whole plate as one
forward pass -- no per-character segmentation or bounding boxes needed.

Why this instead of general OCR (EasyOCR, used elsewhere in evaluation/):
general OCR is trained on arbitrary document/scene text and has to guess
at font, spacing and layout; LPRNet is trained ONLY on plate-shaped crops
with a small fixed alphabet, so it can be much smaller (a few MB, runs in
milliseconds) and, once trained on your actual plate format, much more
accurate than a general-purpose reader.

How it works, in short: the backbone is a stack of small conv blocks that
shrink the image down to a strip of shape (channels, 1, W') -- one column
per few pixels of the original plate width, each column already a rough
"this looks like it might be a digit/letter here" feature vector. That
strip becomes a sequence of length W', and CTC loss (see train_lprnet.py)
lets the network output that sequence WITHOUT knowing in advance how many
characters the plate has or exactly where each one starts -- it just
learns to put a spike of the right character at roughly the right column,
and CTC decoding collapses repeats/blanks into the final text.

Architecture follows the widely-used reference PyTorch port structure
(backbone of small_basic_blocks + maxpools, a global-context branch
pooled and concatenated back in, final 1x1 conv to num_classes channels).

Input size: 200x24 (W x H), not the original paper's 94x24 -- the paper's
default targets ~7-8 character plates and only gives a sequence length
T=10, but this project's ground truth labels run up to 11 characters
(avg 7.2). CTC needs T to exceed the label length, with extra headroom
for any adjacent repeated characters, so the width was widened until T
comfortably clears that (200px -> T=23; see train_lprnet.py's vocab/CTC
setup for why this matters).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class small_basic_block(nn.Module):
    """1x1 -> 3x1 -> 1x3 -> 1x1: a bottleneck that keeps parameter count
    low while still covering a decent receptive field, used throughout
    the backbone instead of plain 3x3 convs."""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        ch_mid = ch_out // 4
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_mid, ch_mid, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_mid, ch_mid, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_mid, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        """num_classes = alphabet size + 1 (the CTC blank token, index 0
        by convention here -- see lprnet_vocab.py)."""
        super().__init__()
        self.num_classes = num_classes

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2))

        self.stage2 = nn.Sequential(
            small_basic_block(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(2, 2))

        self.stage3 = nn.Sequential(
            small_basic_block(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            small_basic_block(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=(2, 2))
        self.dropout1 = nn.Dropout(dropout_rate)

        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(4, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.dropout2 = nn.Dropout(dropout_rate)

        self.classifier = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=(1, 13), padding=(0, 6)),
            nn.BatchNorm2d(num_classes),
        )

        # Global context: pool stage2's features down to one value per
        # channel per column, then add back into the final sequence --
        # gives every output position a cheap summary of the whole plate,
        # not just its local receptive field, which helps disambiguate
        # similar-looking characters using overall plate context.
        self.context_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        context = x
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.dropout1(x)
        x = self.stage4(x)
        x = self.dropout2(x)
        x = self.classifier(x)  # (B, num_classes, H', W')

        # Collapse height to 1 (should already be ~1 by here for a 24px-
        # tall input; average pool handles off-by-one sizes robustly).
        x = F.adaptive_avg_pool2d(x, (1, x.shape[-1]))
        x = x.squeeze(2)  # (B, num_classes, W') -- W' is the sequence length

        ctx = self.context_conv(context)
        ctx = F.adaptive_avg_pool2d(ctx, (1, x.shape[-1]))
        ctx = ctx.squeeze(2)
        x = x + ctx

        return x  # (B, num_classes, T) -- log_softmax applied by the CTC loss caller


if __name__ == "__main__":
    model = LPRNet(num_classes=37)
    dummy = torch.randn(2, 3, 24, 200)  # (B, C, H, W)
    out = model(dummy)
    print("output shape:", out.shape)  # expect (2, 37, 23)
