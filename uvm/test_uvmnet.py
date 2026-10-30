import argparse
import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model_uvmnet import UVMNet
from clahe_preprocess import apply_clahe_pil


# ============================================================
# Configuration
# ============================================================

# Defaults sized so a full 2048x1024 Cityscapes-scale frame processes in a
# SINGLE tile (no seams): probed at ~5.6GB peak (uvmb_size=32, inference,
# no grad) on an 8GB card, well within budget. The old 64x64/32 defaults
# tiled every frame into a ~32x16 grid of tiles stitched back together --
# each tile got its own independent atmospheric-light/haze estimate with
# no shared context, which shows up as visible seams and inconsistent
# dehazing strength across the image. Tiling only kicks in now for images
# too large to fit at once; --tile_size/--overlap remain configurable for
# that case or for smaller GPUs.
TILE_SIZE = 1536
OVERLAP = 128
STRIDE = TILE_SIZE - OVERLAP

# The UNet has 4 MaxPool2d(2) stages (see unet_parts.py), so spatial dims
# must be divisible by 2**4=16 for the encoder/decoder skip connections to
# line up. This is the ONLY real size constraint -- it is NOT tied to
# TILE_SIZE.
ALIGN = 16


# ============================================================
# Tile position helper
# ============================================================

def get_positions(length, tile_size, stride):
    """
    Generate tile start positions so the complete image is covered.
    """

    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))

    # Make sure the final tile reaches the image boundary
    final_position = length - tile_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


# ============================================================
# Pad image to minimum tile size
# ============================================================

def pad_to_tile(image_tensor, tile_size):
    """
    Pad an image so both dimensions are at least tile_size.

    Input:
        [1, 3, H, W]

    Output:
        padded tensor
        original H
        original W
    """

    _, _, h, w = image_tensor.shape

    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)

    if pad_h == 0 and pad_w == 0:
        return image_tensor, h, w

    # Replicate padding works even for very small images.
    padded = F.pad(
        image_tensor,
        (0, pad_w, 0, pad_h),
        mode="replicate"
    )

    return padded, h, w


# ============================================================
# Feathered tile blend weight
# ============================================================

def tile_weight(y, x, tile_size, overlap, padded_h, padded_w, device):
    """
    2D weight mask for one tile: ramps linearly from 0 -> 1 across the
    overlap band on whichever edges border a neighboring tile (not on
    edges that coincide with the padded image boundary), full weight 1.0
    in the interior. Replaces the old uniform weight=1 blend, which
    averaged overlapping tiles but still left a visible seam right at the
    overlap boundary -- each tile gets its own independent atmospheric-
    light/haze estimate (see this file's TILE_SIZE comment), so a hard
    switch from one tile's estimate to the blended average is visible.
    A gradual feather across the whole overlap band hides that instead.
    """
    ramp = torch.linspace(0.0, 1.0, overlap, device=device)

    wy = torch.ones(tile_size, device=device)
    if y > 0:
        wy[:overlap] = ramp
    if y + tile_size < padded_h:
        wy[-overlap:] = torch.minimum(wy[-overlap:], ramp.flip(0))

    wx = torch.ones(tile_size, device=device)
    if x > 0:
        wx[:overlap] = ramp
    if x + tile_size < padded_w:
        wx[-overlap:] = torch.minimum(wx[-overlap:], ramp.flip(0))

    return (wy.view(-1, 1) * wx.view(1, -1)).clamp(min=1e-3)


# ============================================================
# Process one image
# ============================================================

@torch.no_grad()
def process_image(model, image, device):
    """
    Process an arbitrary-resolution image using overlapping
    64x64 tiles.

    The output has exactly the same dimensions as the input.
    """

    # --------------------------------------------------------
    # Convert PIL image -> tensor
    # --------------------------------------------------------

    to_tensor = transforms.ToTensor()

    image_tensor = to_tensor(image).unsqueeze(0).to(device)

    _, _, original_h, original_w = image_tensor.shape

    print(f"    Input size: {original_w} x {original_h}")

    # --------------------------------------------------------
    # Small-image fast path: pad only up to the network's actual
    # alignment requirement (multiple of 16), NOT the full TILE_SIZE.
    #
    # DoubleConv resizes every feature map to a fixed uvmb_size x
    # uvmb_size grid before running its SSM (see unet_parts.py), at
    # EVERY level of the network, including right at the input. Padding
    # a small image (e.g. 64x64) up to a 1536x1536 canvas -- 99.8%
    # replicated padding -- means that fixed-size resize squashes the
    # real content down to roughly a single pixel of the SSM's grid, so
    # the model's scene understanding is almost entirely padding and it
    # applies almost no correction. Keeping the padded canvas close to
    # the image's own size avoids that dilution entirely.
    # --------------------------------------------------------

    if original_h <= TILE_SIZE and original_w <= TILE_SIZE:
        target_h = ((original_h + ALIGN - 1) // ALIGN) * ALIGN
        target_w = ((original_w + ALIGN - 1) // ALIGN) * ALIGN
        padded = F.pad(
            image_tensor,
            (0, target_w - original_w, 0, target_h - original_h),
            mode="replicate"
        )
        print(f"    Processing 1 tile(s) ({target_w}x{target_h}, native-sized, no OVERLAP)")
        prediction = torch.clamp(model(padded), 0.0, 1.0)
        output = prediction[:, :, :original_h, :original_w].squeeze(0).cpu()
        output_image = transforms.ToPILImage()(output)
        print(f"    Output size: {output_image.width} x {output_image.height}")
        return output_image

    # --------------------------------------------------------
    # Pad if image is smaller than one tile
    # --------------------------------------------------------

    padded, _, _ = pad_to_tile(
        image_tensor,
        TILE_SIZE
    )

    _, _, padded_h, padded_w = padded.shape

    # --------------------------------------------------------
    # Generate tile positions
    # --------------------------------------------------------

    y_positions = get_positions(
        padded_h,
        TILE_SIZE,
        STRIDE
    )

    x_positions = get_positions(
        padded_w,
        TILE_SIZE,
        STRIDE
    )

    num_tiles = len(y_positions) * len(x_positions)

    print(
        f"    Processing {num_tiles} tile(s) "
        f"({TILE_SIZE}x{TILE_SIZE}, overlap={OVERLAP})"
    )

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    output_sum = torch.zeros_like(padded)

    weight_sum = torch.zeros(
        (1, 1, padded_h, padded_w),
        device=device
    )

    # --------------------------------------------------------
    # Process tiles
    # --------------------------------------------------------

    for y in y_positions:

        for x in x_positions:

            tile = padded[
                :,
                :,
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            # Safety check
            if tile.shape[-2:] != (TILE_SIZE, TILE_SIZE):
                raise RuntimeError(
                    f"Invalid tile size: {tile.shape}"
                )

            # ------------------------------------------------
            # UVM-Net inference
            # ------------------------------------------------

            prediction = model(tile)

            # Keep valid image range
            prediction = torch.clamp(
                prediction,
                0.0,
                1.0
            )

            # ------------------------------------------------
            # Accumulate (feathered weight, not uniform +=1 --
            # see tile_weight()'s docstring for why)
            # ------------------------------------------------

            w = tile_weight(y, x, TILE_SIZE, OVERLAP, padded_h, padded_w, device)

            output_sum[
                :,
                :,
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ] += prediction * w

            weight_sum[
                :,
                :,
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ] += w

    # --------------------------------------------------------
    # Average overlapping predictions
    # --------------------------------------------------------

    output = output_sum / torch.clamp(
        weight_sum,
        min=1e-8
    )

    # --------------------------------------------------------
    # Remove padding
    # --------------------------------------------------------

    output = output[
        :,
        :,
        :original_h,
        :original_w
    ]

    output = torch.clamp(
        output,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Tensor -> PIL
    # --------------------------------------------------------

    output = output.squeeze(0).cpu()

    output_image = transforms.ToPILImage()(output)

    print(
        f"    Output size: {output_image.width} x "
        f"{output_image.height}"
    )

    return output_image


# ============================================================
# Find images
# ============================================================

def get_image_files(input_path):

    extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff"
    )

    if os.path.isfile(input_path):
        return [input_path]

    image_files = []

    # Recurse -- Foggy_Driving in particular nests images under
    # record-type subfolders (leftImg8bit/test/{pedestrian,public,web}/),
    # not flat in the split directory itself.
    for root, _dirs, names in os.walk(input_path):
        for name in sorted(names):
            if name.lower().endswith(extensions):
                image_files.append(os.path.join(root, name))

    return sorted(image_files)


# ============================================================
# Main
# ============================================================

def main():

    global TILE_SIZE, OVERLAP, STRIDE

    parser = argparse.ArgumentParser(
        description="Adaptive tiled inference for UVM-Net"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input image or image directory"
    )

    parser.add_argument(
        "--output",
        default="test_outputs",
        help="Output directory"
    )

    parser.add_argument(
        "--checkpoint",
        default="uvmnet_ckpt.pt",
        help="UVM-Net checkpoint"
    )

    parser.add_argument(
        "--device",
        default="cuda"
    )

    parser.add_argument(
        "--backend",
        default="cuda"
    )

    parser.add_argument(
        "--uvmb_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--tile_size",
        type=int,
        default=TILE_SIZE,
        help="Max tile size for one forward pass. Default covers a full "
             "2048x1024 frame in a single tile (~5.6GB peak at uvmb_size=32); "
             "lower this if you hit OOM on a smaller GPU or larger images."
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
        help="Overlap between tiles when tiling is actually needed "
             "(image larger than --tile_size)."
    )

    parser.add_argument(
        "--clahe",
        action="store_true",
        default=False,
        help="Apply CLAHE to the input before running it through the model "
             "(see clahe_preprocess.py). Must match whatever the checkpoint "
             "was trained with -- pass this if you trained with --clahe, "
             "leave it off otherwise."
    )

    args = parser.parse_args()

    TILE_SIZE = args.tile_size
    OVERLAP = args.overlap
    STRIDE = TILE_SIZE - OVERLAP

    # --------------------------------------------------------
    # Check CUDA
    # --------------------------------------------------------

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but CUDA is not available."
        )

    device = torch.device(args.device)

    print("=" * 60)
    print("UVM-Net Adaptive Tiled Inference")
    print("=" * 60)

    print(f"Device      : {device}")
    print(f"Backend     : {args.backend}")
    print(f"UVMB size   : {args.uvmb_size}")
    print(f"Tile size   : {TILE_SIZE}")
    print(f"Overlap     : {OVERLAP}")
    print(f"Checkpoint  : {args.checkpoint}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading model...")

    model = UVMNet(
        n_channels=3,
        backend=args.backend,
        uvmb_size=args.uvmb_size
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_files = get_image_files(
        args.input
    )

    if not image_files:
        raise RuntimeError(
            f"No images found in: {args.input}"
        )

    print()
    print(f"Found {len(image_files)} image(s).")

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        args.output,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(image_files)}] "
            f"{os.path.basename(image_path)}"
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            if args.clahe:
                image = apply_clahe_pil(image)

            output_image = process_image(
                model,
                image,
                device
            )

            filename = os.path.basename(
                image_path
            )

            stem, _ = os.path.splitext(
                filename
            )

            output_path = os.path.join(
                args.output,
                stem + "_restored.png"
            )

            output_image.save(
                output_path
            )

            print(
                f"    Saved: {output_path}"
            )

        except Exception as e:

            print(
                f"    ERROR: {e}"
            )

        finally:
            # Release cached blocks between images -- without this, OOM
            # retries on one large image fragment the allocator and can
            # starve the next image even though "free" memory looks
            # sufficient (observed with a 2560x1707 image in a batch of
            # otherwise small crops).
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("Inference complete.")
    print(f"Results: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()