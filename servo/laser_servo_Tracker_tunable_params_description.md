# Laser-Servo Control Tuning Reference

| Parameter              | What it controls                                                                 | Effect of increasing                          | Effect of decreasing                     |
|-------------------------|------------------------------------------------------------------------------------|------------------------------------------------|--------------------------------------------|
| `PIXEL_TO_US` (Kp)       | Proportional gain — how many microseconds of servo pulse change per pixel of error | Faster response, but more overshoot/oscillation | Slower response, more stable, may feel sluggish |
| `PIXEL_TO_US_D` (Kd, new)| Derivative gain — dampens correction based on how fast error is changing          | More damping, less overshoot; too high = sluggish/jittery | Less damping, more overshoot |
| `MAX_STEP_US`            | Hard cap on pulse change per control update                                       | Faster max slew rate, risk of overshoot         | Safer, slower max response                |
| `PIXEL_DEADBAND`         | Error (px) below which the system considers target "reached" and stops correcting | Less dithering near target, less precision      | More precision, more risk of noise-driven micro-oscillation |
| `SERVO_UPDATE_INTERVAL`  | Minimum time between servo corrections                                            | Gives servo more time to physically settle before next correction (reduces lag-induced oscillation) | Faster reaction, but risk of commanding moves before the previous one finished |
| `ALPHA` (new, EMA filter)| Smoothing factor for laser position readings                                      | Smoother/lagged tracking, less noise sensitivity | Twitchier, more responsive to raw noise   |

## Tuning order
1. Set `PIXEL_TO_US_D` to ~0 and `SERVO_UPDATE_INTERVAL` generous (e.g. 0.2s). Confirm it settles without oscillating (even if slow).
2. Gradually reduce `SERVO_UPDATE_INTERVAL` until oscillation reappears, then back off slightly.
3. Increase `PIXEL_TO_US` for faster response until you see overshoot, then add `PIXEL_TO_US_D` to damp it out.
4. Fine-tune `PIXEL_DEADBAND` and `ALPHA` last, based on how noisy your laser detection looks in practice.