# Laser-Servo Controller — New Variables Reference

## Detection continuity (fixes the CSV-observed oscillation)

The CSV showed `laser_x` alternating between two real, similarly-sized bright
blobs (likely the laser dot and a reflection) while the target stayed fixed.
Picking "largest blob" each frame let the detector flip between them, which
is what actually caused the servo to swing back and forth.

| Variable | What it controls | Increase it | Decrease it |
|---|---|---|---|
| `MAX_JUMP_PX` | Max frame-to-frame centroid movement (px) accepted as "the same blob." A farther candidate is ignored for that frame. | More tolerant of fast genuine laser motion; more likely to let the reflection-jump problem back in. | Stricter — rejects more false jumps, but may also reject real fast motion. |
| `REACQUIRE_TIMEOUT` | Seconds with no accepted detection before gating is dropped and the next candidate is accepted regardless. | Slower to recover if tracking gets stuck on a stale/wrong blob. | Faster recovery, but more chances to re-lock onto the wrong blob if both are still visible. |

**Caveat worth knowing:** this only prevents *switching* between blobs once
one is locked — it doesn't guarantee the *first* acquisition (when there's no
prior position, `last_x is None`) picks the real laser over the reflection,
since that first pick still falls back to "largest area." If tracking
consistently locks onto the wrong blob from the start, the more robust fix
is tightening `BRIGHTNESS_MIN`/`MIN_BLOB_AREA` (from the earlier reference)
so the reflection doesn't pass the detector at all, or physically reducing
the reflective surface in view.

## Adam-style adaptive control

| Variable | What it controls | Increase it | Decrease it |
|---|---|---|---|
| `ADAM_LR` | Base learning rate — overall correction strength (replaces the old flat `PIXEL_TO_US`). | Faster overall response. | Slower, gentler response. |
| `ADAM_BETA1` | Momentum — how much recent error history smooths each step vs. reacting to the instantaneous error. | Smoother, more resistant to single-frame noise/jump spikes; slower to change direction. | Snappier but more reactive to noise. |
| `ADAM_BETA2` | Controls the running estimate of error variance, used to normalize step size. | Slower-adapting normalization (steadier steps over longer windows). | Faster-adapting normalization (reacts to recent noise level quickly). |
| `ADAM_EPS` | Small constant preventing divide-by-zero when variance is near zero. | Damps steps more when error has been very consistent/small. | Allows sharper steps in that same situation. Rarely needs tuning. |

**Important nuance:** Adam by itself does *not* guarantee smaller steps as
you approach the target — it normalizes by the gradient's own recent
magnitude, so near a stable target it can keep taking similarly-sized steps
rather than shrinking to zero. The momentum/normalization here is mainly for
**smoothing against noisy or jumpy error readings** (like the CSV's
detection jumps). The actual "take smaller steps as we get closer" behavior
comes from the easing term below.

| Variable | What it controls | Increase it | Decrease it |
|---|---|---|---|
| `EASE_DISTANCE_PX` | Distance (px) at which step size starts scaling down toward the target. At/beyond this distance: full-strength step. At 0 error: step scaled to 0. Linear in between. | Slower approach speed near the target, but smoother final settling. | Faster approach near the target, but more prone to overshoot right at the end. |
| `MAX_STEP_US` | Hard safety cap on any single correction regardless of what Adam/easing compute. | Faster max response to large errors. | Safer, slower worst-case response. |

## Post-target jitter

| Variable | What it controls | Increase it | Decrease it |
|---|---|---|---|
| `JITTER_ENABLE` | Turns the jitter behavior on/off entirely. | — | — |
| `JITTER_AMPLITUDE_US` | How far (in microseconds) the pulse swings from the settled position. | More visible/aggressive wiggle. | Subtler wiggle. |
| `JITTER_DURATION_S` | How long (seconds) the jitter runs after the target is first reached. | Longer wiggle period. | Shorter, or effectively a quick "twitch." |
| `JITTER_FREQUENCY_HZ` | Speed of the sinusoidal component of the motion. | Faster wiggling. | Slower, more lazy wiggling. |
| `JITTER_RANDOMNESS` | Blend between smooth sine motion (`0.0`) and pure random noise (`1.0`). | More erratic/organic-looking jitter. | More mechanical, predictable wiggle. |

After `JITTER_DURATION_S` elapses, the servo returns to the exact pulse it
had when the target was first reached and holds still (`SETTLED` state) —
selecting a new target (click) or pressing `C`/`R` resets this state machine.

## Extra diagnostic log

A new file, `<name>_extra.csv`, is written alongside your existing log and
video (neither of which is modified). It logs, every frame: the raw
(pre-filter, pre-gating) laser position, whether a candidate was rejected as
an implausible jump and by how much, the filtered position actually used for
control, the live Adam momentum/variance/step values, the easing factor,
the final correction applied, current jitter offset, and the exact on-screen
status text — so future debugging doesn't require pulling frames out of a
video or OCR'ing overlay text again.