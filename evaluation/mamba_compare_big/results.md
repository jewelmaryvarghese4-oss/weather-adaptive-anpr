# Mamba dehazing checkpoint comparison

| config | n | PSNR | SSIM | Laplacian var (native) | Laplacian var (norm128) | OCR conf | OCR exact match | OCR char acc |
|---|---|---|---|---|---|---|---|---|
| baseline_hazy_128 | 420 | 14.34 | 0.7009 | 308.4 | 308.4 | 0.3119 | 0.0214 | 0.1749 |
| ckpt128_epoch40 | 420 | 26.62 | 0.9266 | 866.4 | 866.4 | 0.3622 | 0.0310 | 0.2056 |
| ckpt128_epoch50 | 420 | 26.84 | 0.9274 | 880.7 | 880.7 | 0.3656 | 0.0286 | 0.1991 |
| ckpt128_perc035_epoch50 | 420 | 26.90 | 0.9348 | 866.6 | 866.6 | 0.3812 | 0.0310 | 0.1962 |
