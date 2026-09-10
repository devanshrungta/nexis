#!/usr/bin/env python3

import time

# ============================================================
# PWM
# ============================================================

PWM_CHIP = "/sys/class/pwm/pwmchip3"
PWM_PATH = "/sys/class/pwm/pwmchip3/pwm0"
PWM_CHANNEL = 0

PERIOD_NS = 20_000_000       # 20 ms = 50 Hz


# ============================================================
# SERVO
# ============================================================

MIN_PULSE_US = 500
MAX_PULSE_US = 2500

TRAVEL_TIME_S = 3.0          # 3 seconds from one extreme to the other
UPDATE_INTERVAL_S = 0.02     # 50 updates/sec


# ============================================================
# PWM FUNCTIONS
# ============================================================

def write_sysfs(path, value):
    with open(path, "w") as f:
        f.write(str(value))


def set_pulse_us(pulse_us):
    pulse_us = max(MIN_PULSE_US, min(MAX_PULSE_US, pulse_us))

    pulse_ns = int(pulse_us * 1000)

    write_sysfs(
        f"{PWM_PATH}/duty_cycle",
        pulse_ns
    )


def setup_pwm():

    if not __import__("os").path.exists(PWM_PATH):
        write_sysfs(
            f"{PWM_CHIP}/export",
            PWM_CHANNEL
        )
        time.sleep(0.1)

    try:
        write_sysfs(
            f"{PWM_PATH}/enable",
            0
        )
    except Exception:
        pass

    write_sysfs(
        f"{PWM_PATH}/period",
        PERIOD_NS
    )

    set_pulse_us(MIN_PULSE_US)

    write_sysfs(
        f"{PWM_PATH}/enable",
        1
    )


# ============================================================
# SERVO OSCILLATION
# ============================================================

def move_servo(start_us, end_us):

    start_time = time.monotonic()

    while True:

        elapsed = time.monotonic() - start_time

        # 0.0 -> 1.0 over exactly TRAVEL_TIME_S
        progress = elapsed / TRAVEL_TIME_S

        if progress >= 1.0:
            progress = 1.0

        # Linear interpolation
        pulse_us = (
            start_us
            + (end_us - start_us) * progress
        )

        set_pulse_us(pulse_us)

        if progress >= 1.0:
            break

        time.sleep(UPDATE_INTERVAL_S)


# ============================================================
# MAIN
# ============================================================

def main():

    print("DS3240MG servo oscillation")
    print(f"Pulse range : {MIN_PULSE_US} - {MAX_PULSE_US} us")
    print(f"Travel time : {TRAVEL_TIME_S} seconds")
    print()
    print("Press Ctrl+C to stop.")
    print()

    setup_pwm()

    try:

        while True:

            # 500 -> 2500 us in 3 seconds
            move_servo(
                MIN_PULSE_US,
                MAX_PULSE_US
            )

            # 2500 -> 500 us in 3 seconds
            move_servo(
                MAX_PULSE_US,
                MIN_PULSE_US
            )

    except KeyboardInterrupt:

        print("\nStopping servo...")

        # Leave servo at neutral
        set_pulse_us(1500)

        time.sleep(0.5)


if __name__ == "__main__":
    main()
