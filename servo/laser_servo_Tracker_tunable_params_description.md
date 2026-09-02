# Laser-Servo Controller — New Variables Reference

## Detection

The nearest-blob continuity gating (`MAX_JUMP_PX`, `REACQUIRE_TIMEOUT`) has
been **removed** — it was rejecting valid detections and occasionally
locking onto the wrong blob. Detection is back to your original
"largest bright blob wins" logic, unchanged. If the two-blob
(laser + reflection) issue seen in the CSV comes back, the more reliable
fix is tightening `BRIGHTNESS_MIN`/`MIN_BLOB_AREA` so the reflection
doesn't pass the detector at all, or physically addressing the reflective
surface — not re-adding detection-side gating.

## State machine fix

Previously, once the controller reached `SETTLED`, it stopped checking the
real laser position entirely — there was no path back to `TRACKING`. A
single bad reading that briefly looked "within deadband" (e.g. from a
detection glitch) would lock the display on `TARGET REACHED` forever, even
while the real laser sat elsewhere. `SETTLED` now keeps re-evaluating error
on every throttled tick, and resumes `TRACKING`/correcting automatically if
the error drifts back outside `PIXEL_DEADBAND`.

## Adam-style adaptive control

| Variable | What it controls | Increase it | Decrease it |
|---|---|---|---|
| `ADAM_LR` | Direct step size in microseconds on large errors — **not** a per-pixel gain like the old `PIXEL_TO_US`. After the first update, Adam's normalized step settles to roughly `± ADAM_LR`, regardless of error magnitude, until `EASE_DISTANCE_PX` scales it down near the target. | Faster response on large errors. | Slower — too low (e.g. sub-1us) and the servo effectively won't move at all, since the resulting duty-cycle change is below what the hardware can act on. |
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
video (neither of which is modified). It logs, every frame: the detected
laser position, the filtered position actually used for control, the live
Adam momentum/variance/step values, the easing factor, the final correction
applied, current jitter offset, the controller's state (`TRACKING`/
`JITTERING`/`SETTLED`), and the exact on-screen status text — so future
debugging doesn't require pulling frames out of a video or OCR'ing overlay
text again.