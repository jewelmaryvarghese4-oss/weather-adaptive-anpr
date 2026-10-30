from __future__ import annotations
import numpy as np
import cv2

from dehaze_baseline import (
    compute_dark_channel,
    estimate_transmission,
    refine_transmission_guided,
    recover_radiance,
    apply_gamma_correction,
    apply_white_balance,
)


# ---------------------------------------------------------------------------
# Spec 2.1 - Glare / specular-highlight mask
# ---------------------------------------------------------------------------
def build_glare_mask(image: np.ndarray, v_thresh: int = 240,
                      s_thresh: int = 30) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s_channel, v_channel = cv2.split(hsv)
    glare_mask = (v_channel > v_thresh) & (s_channel < s_thresh)
    return glare_mask


# ---------------------------------------------------------------------------
# Spec 1.3 (modified) - Atmospheric light with glare exclusion + a_max_cap
# ---------------------------------------------------------------------------
def estimate_atmospheric_light_v2(image: np.ndarray, dark_channel: np.ndarray,
                                   candidate_mask: np.ndarray | None = None,
                                   omega: float = 0.15, alpha: float = 0.8,
                                   top_fraction: float = 0.001,
                                   a_max_cap: float = 140.0) -> np.ndarray:
    h, w = dark_channel.shape
    num_pixels = h * w

    flat_dark = dark_channel.reshape(num_pixels).astype(np.float64)
    flat_img = image.reshape(num_pixels, 3).astype(np.float64)

    if candidate_mask is not None:
        flat_mask = candidate_mask.reshape(num_pixels)
        if flat_mask.any():
            flat_dark = np.where(flat_mask, flat_dark, -np.inf)

    num_top = max(int(num_pixels * top_fraction), 1)
    top_indices = np.argpartition(flat_dark, -num_top)[-num_top:]

    top_pixels = flat_img[top_indices]
    atmospheric_light = alpha * top_pixels.mean(axis=0)
    atmospheric_light = np.minimum(atmospheric_light, a_max_cap)
    return atmospheric_light


# ---------------------------------------------------------------------------
# Spec 2.2 - Full v2 pipeline
# ---------------------------------------------------------------------------
def dehaze_frame_v2(image: np.ndarray, mode: str = "video",
                     omega: float = 0.15, t_min: float = 0.35,
                     alpha: float = 0.8, gamma: float = 1.3,
                     patch_size: int = 15, top_fraction: float = 0.001,
                     guided_radius: int = 40, guided_eps: float = 1e-3,
                     resize_to: tuple[int, int] | None = (640, 480),
                     glare_v_thresh: int = 240, glare_s_thresh: int = 30,
                     a_max_cap: float = 140.0,
                     white_balance: bool = False,
                     brightness_skip_gamma_thresh: float = 140.0,
                     return_glare_mask: bool = False):
    if mode not in ("video", "image"):
        raise ValueError("mode must be 'video' or 'image'")

    original_size = (image.shape[1], image.shape[0])
    working_image = image
    if mode == "video" and resize_to is not None:
        working_image = cv2.resize(image, resize_to, interpolation=cv2.INTER_LINEAR)

    glare_mask = build_glare_mask(working_image, glare_v_thresh, glare_s_thresh)
    candidate_mask = ~glare_mask
    if not candidate_mask.any():
        candidate_mask = None

    dark_channel = compute_dark_channel(working_image, patch_size)
    atmospheric_light = estimate_atmospheric_light_v2(
        working_image, dark_channel, candidate_mask=candidate_mask,
        omega=omega, alpha=alpha, top_fraction=top_fraction, a_max_cap=a_max_cap
    )
    transmission = estimate_transmission(working_image, atmospheric_light, omega, t_min)

    if mode == "image":
        transmission = refine_transmission_guided(
            working_image, transmission, guided_radius, guided_eps
        )
        transmission = np.maximum(transmission, t_min)

    recovered = recover_radiance(working_image, atmospheric_light, transmission)

    mean_brightness = cv2.cvtColor(recovered, cv2.COLOR_BGR2GRAY).mean()
    if mean_brightness < brightness_skip_gamma_thresh:
        recovered = apply_gamma_correction(recovered, gamma)

    if white_balance:
        recovered = apply_white_balance(recovered)

    if mode == "video" and resize_to is not None:
        recovered = cv2.resize(recovered, original_size, interpolation=cv2.INTER_LINEAR)

    if return_glare_mask:
        return recovered, glare_mask
    return recovered


# ---------------------------------------------------------------------------
# Test block - baseline vs v2 comparison
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dehaze_baseline import dehaze_frame

    img = cv2.imread("../datasets/dataset/hazy/crop_0000_v0.jpg")
    if img is None:
        print("Couldn't load image — check the filename/path")
    else:
        print("Image shape:", img.shape)

        out_baseline = dehaze_frame(img, mode="video", resize_to=None)
        cv2.imwrite("results/compare_baseline.jpg", out_baseline)

        out_v2, glare_mask = dehaze_frame_v2(
            img, mode="video", resize_to=None, return_glare_mask=True
        )
        cv2.imwrite("results/compare_v2.jpg", out_v2)

        print("Glare pixels flagged:", glare_mask.sum(), "out of", glare_mask.size)
        cv2.imwrite("results/glare_mask_debug.jpg", (glare_mask.astype(np.uint8) * 255))

        print("Saved results/compare_baseline.jpg, results/compare_v2.jpg, results/glare_mask_debug.jpg")