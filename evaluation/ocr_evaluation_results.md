# OCR Confidence Evaluation Results

Full dataset: 1968 images

## Full dataset

| Version | Avg. confidence | Images scored | No-text images |
|---|---|---|---|
| Clear ground truth | 0.5169 | 1584 | 384 |
| Hazy (untouched) | 0.5019 | 1320 | 648 |
| Dehazed (DCP baseline) | 0.5196 | 1332 | 636 |
| Dehazed (DCP + glare-aware, v2) | 0.5173 | 1329 | 639 |

## Glare-heavy subset (340 images)

| Version | Avg. confidence | Images scored | No-text images |
|---|---|---|---|
| Clear ground truth | 0.5732 | 306 | 34 |
| Hazy (untouched) | 0.5516 | 280 | 60 |
| Dehazed (DCP baseline) | 0.5701 | 283 | 57 |
| Dehazed (DCP + glare-aware, v2) | 0.5537 | 279 | 61 |
