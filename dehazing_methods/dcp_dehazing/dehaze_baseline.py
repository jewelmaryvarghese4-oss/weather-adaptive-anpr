from __future__ import annotations
import time
import numpy as np
import cv2

try:
    import cv2.ximgproc as _ximgproc
    _HAS_XIMGPROC = True
except ImportError:
    _HAS_XIMGPROC = False

try:
    import cv2.xphoto as _xphoto
    _HAS_XPHOTO = True
except ImportError:
    _HAS_XPHOTO = False


# ---------------------------------------------------------------------------
# Spec 1.2 - Dark channel computation, Eq. (2)
# ---------------------------------------------------------------------------
def compute_dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    channel_min = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(channel_min, kernel)
    return dark_channel


# ---------------------------------------------------------------------------
# Spec 1.3 - Atmospheric light with the stability fix, Eq. (3)
# Includes a_max_cap: prevents A from locking onto near-pure-white glare/sky
# pixels in full road/vehicle scenes, which was causing severe overexposure.
# Default a_max_cap=140.0 was tuned against bright-sky/glare-heavy scenes.
# ---------------------------------------------------------------------------
def estimate_atmospheric_light(image: np.ndarray, dark_channel: np.ndarray,
                                omega: float = 0.15, alpha: float = 0.8,
                                top_fraction: float = 0.001,
                                a_max_cap: float = 140.0) -> np.ndarray:
    h, w = dark_channel.shape
    num_pixels = h * w

    flat_dark = dark_channel.reshape(num_pixels).astype(np.float64)
    flat_img = image.reshape(num_pixels, 3).astype(np.float64)

    num_top = max(int(num_pixels * top_fraction), 1)
    top_indices = np.argpartition(flat_dark, -num_top)[-num_top:]

    top_pixels = flat_img[top_indices]
    atmospheric_light = alpha * top_pixels.mean(axis=0)
    atmospheric_light = np.minimum(atmospheric_light, a_max_cap)
    return atmospheric_light


# ---------------------------------------------------------------------------
# Spec 1.4 - Fast transmission approximation, Eq. (4)-(5)
# Default omega=0.15, t_min=0.35: tuned down from the paper's 0.5/0.05 to
# reduce over-amplification on bright sky/road-glare pixels (see tuning
# notes - the original defaults caused severe blown-out overexposure on
# full road-scene images, unlike the tightly-cropped plate images this
# was first tested on).
# ---------------------------------------------------------------------------
def estimate_transmission(image: np.ndarray, atmospheric_light: np.ndarray,
                           omega: float = 0.15, t_min: float = 0.35) -> np.ndarray:
    channel_min = np.min(image, axis=2).astype(np.float64)
    a_max = max(np.max(atmospheric_light), 1e-6)
    t_hat = 1.0 - omega * (channel_min / a_max)
    transmission = np.maximum(t_hat, t_min)
    return transmission


# ---------------------------------------------------------------------------
# Spec 1.5 - Guided filter refinement, Eq. (6). IMAGE MODE ONLY.
# ---------------------------------------------------------------------------
def refine_transmission_guided(image: np.ndarray, transmission: np.ndarray,
                                radius: int = 40, eps: float = 1e-3) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = transmission.astype(np.float32)

    if _HAS_XIMGPROC:
        return _ximgproc.guidedFilter(gray, t, radius, eps)

    mean_I = cv2.boxFilter(gray, -1, (radius, radius))
    mean_p = cv2.boxFilter(t, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(gray * t, -1, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(gray * gray, -1, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    return mean_a * gray + mean_b


# ---------------------------------------------------------------------------
# Spec 1.6 - Radiance recovery, Eq. (7)
#   J_c(x) = (I_c(x) - A_c) / t(x) + A_c
# ---------------------------------------------------------------------------
def recover_radiance(image: np.ndarray, atmospheric_light: np.ndarray,
                      transmission: np.ndarray) -> np.ndarray:
    img_f = image.astype(np.float64)
    t = transmission.astype(np.float64)
    recovered = np.empty_like(img_f)
    for c in range(3):
        recovered[:, :, c] = (img_f[:, :, c] - atmospheric_light[c]) / t + atmospheric_light[c]
    return np.clip(recovered, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Spec 1.7 - Post-processing: gamma correction, Eq. (8) + optional white balance
# ---------------------------------------------------------------------------
def apply_gamma_correction(image: np.ndarray, gamma: float = 1.3) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                       for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def apply_white_balance(image: np.ndarray) -> np.ndarray:
    if _HAS_XPHOTO:
        return _xphoto.createSimpleWB().balanceWhite(image)
    return image


# ---------------------------------------------------------------------------
# Spec 1.8 - Full pipeline, two operating modes
# Fix applied: recover_radiance() now runs unconditionally (was accidentally
# nested inside the "if mode == 'image'" block, causing UnboundLocalError in
# video mode). Gamma correction is skipped when the recovered image is
# already bright, to avoid compounding overexposure on sky/glare-heavy scenes.
# a_max_cap is now accepted here and passed through to atmospheric light
# estimation (previously only reachable by calling estimate_atmospheric_light
# directly - dehaze_frame() silently used its own default before this fix).
# ---------------------------------------------------------------------------
def dehaze_frame(image: np.ndarray, mode: str = "video",
                  omega: float = 0.15, t_min: float = 0.35,
                  alpha: float = 0.8, gamma: float = 1.3,
                  patch_size: int = 15, top_fraction: float = 0.001,
                  guided_radius: int = 40, guided_eps: float = 1e-3,
                  resize_to: tuple[int, int] | None = (640, 480),
                  a_max_cap: float = 140.0,
                  white_balance: bool = False,
                  brightness_skip_gamma_thresh: float = 140.0) -> np.ndarray:
    if mode not in ("video", "image"):
        raise ValueError("mode must be 'video' or 'image'")

    original_size = (image.shape[1], image.shape[0])  # (width, height)
    working_image = image

    if mode == "video" and resize_to is not None:
        working_image = cv2.resize(image, resize_to, interpolation=cv2.INTER_LINEAR)

    dark_channel = compute_dark_channel(working_image, patch_size)
    atmospheric_light = estimate_atmospheric_light(
        working_image, dark_channel, omega=omega, alpha=alpha,
        top_fraction=top_fraction, a_max_cap=a_max_cap
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
    # else: image is already bright - skip gamma to avoid compounding overexposure

    if white_balance:
        recovered = apply_white_balance(recovered)

    if mode == "video" and resize_to is not None:
        recovered = cv2.resize(recovered, original_size, interpolation=cv2.INTER_LINEAR)

    return recovered


# ---------------------------------------------------------------------------
# Test block - always keep this LAST in the file
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    img = cv2.imread("../datasets/dataset/hazy/crop_0000_v0.jpg")
    if img is None:
        print("Couldn't load image — check the filename/path")
    else:
        print("Image shape:", img.shape)

        out_video = dehaze_frame(img, mode="video", resize_to=None)
        cv2.imwrite("results/output_video_mode.jpg", out_video)

        out_image = dehaze_frame(img, mode="image")
        cv2.imwrite("results/output_image_mode.jpg", out_image)

        print("Saved output_video_mode.jpg and output_image_mode.jpg")