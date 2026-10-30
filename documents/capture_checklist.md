# Capture checklist: new foggy / rainy plate data

**Goal:** 30+ different cars, each with a plate number you wrote down, photographed in clear weather AND in fog or rain from the same spot.
That gives (1) true plate labels without anyone squinting at blurry crops, (2) real clear/foggy pairs for training the dehazer, and
(3) a fresh test set the models have never seen.

## Why this is needed (what we measured)
- The Sep 26 session has only ~40 distinct readable plates; 34 are already our test set. Video frames added 1 new plate.
- Plates in the videos are 60 to 120 px wide and blurred by compression; the reader works best at 150+ px.
- Fake fog does not match real fog (rain on glass, grain), and the dehazer has never seen real aligned pairs.

## Before you go
- [ ] Phone or camera on a **tripod or fixed stand** (no hand-held shots for the pairs).
- [ ] Lens wiped. **Shoot through open air, not through a windshield or window** (wipers and drops ruin the plate).
- [ ] Turn OFF beauty / HDR / night mode / "AI enhance" / digital zoom. Use the normal main camera.
- [ ] Lock focus and exposure on the car (tap and hold) so brightness does not jump between shots.
- [ ] Full battery, empty storage, and the capture log (`capture_log_template.csv`) ready.

## What to shoot for EACH car
1. **Stand 4 to 10 m behind the car**, camera roughly at plate height, plate facing the camera within about 30 degrees.
   The plate should be **at least 150 px wide in the photo** (about 1/10 of the frame width on a 4000 px photo, more on video).
2. **5 photos** (full resolution), slightly different distances. Keep the car in the same position in the frame.
3. **One 5-second video, 1080p at 60 fps**, steady.
4. **Write the plate number in the log** (copy it from the plate while standing next to the car; mark `?` for any character you are unsure of).
5. Note the **time and conditions** (clear / light fog / dense fog / rain / drizzle on lens / dusk).

## Which cars and conditions
- [ ] At least **30 different cars**, mixed: white, black, silver; single-line and two-line plates; bikes and autos too.
- [ ] **Clear weather first** (same cars, same spot), then **again in fog or rain**. Mark the camera position (a chalk X or a photo of where the tripod stands) so you can return to the exact spot.
- [ ] If a car leaves before the fog arrives, still shoot it in whatever weather you get, and note it. Mixed is fine; every car needs its plate number.
- [ ] Keep **10 cars aside as a test set** (note their IDs in the log). They are never used for training.
- [ ] Include at least one session each of: light fog, dense fog, rain, dusk or low light.

## After shooting
- [ ] Copy everything to one folder named by date (for example `capture_2026-11-02`). Do not edit, crop, rotate or filter the files.
- [ ] Fill in the log: `car_id, plate_text, date, time, condition, test_car (yes/no), filenames, notes`.
- [ ] Send me the folder and the log. I will check the labels against the photos and show you a check sheet before any training.

## Good to know
- Plate numbers are personal data. Keep the folder inside the project and do not post it online.
- Quality beats quantity: 30 cars with a correct plate number and a sharp, close photo are worth more than 300 blurry crops.
