#!/usr/bin/env python3

import os
import sys
import time

PWM_CHIP = "/sys/class/pwm/pwmchip3"
PWM_CHANNEL = 0
PWM_PATH = f"{PWM_CHIP}/pwm{PWM_CHANNEL}"

PERIOD_US = 20000

def write_file(path, value):
    with open(path, "w") as f:
        f.write(str(value))

def setup():

    if not os.path.exists(PWM_PATH):
        print(f"Exporting PWM channel {PWM_CHANNEL}")
        write_file(
            f"{PWM_CHIP}/export",
            PWM_CHANNEL
        )
        time.sleep(0.2)

    try:
        write_file(
            f"{PWM_PATH}/enable",
            0
        )
    except Exception:
        pass

    write_file(
        f"{PWM_PATH}/period",
        PERIOD_US * 1000
    )

    write_file(
        f"{PWM_PATH}/duty_cycle",
        1500 * 1000
    )

    write_file(
        f"{PWM_PATH}/enable",
        1
    )

def set_pulse_us(pulse_us):

    duty_ns = int(round(pulse_us * 1000))

    write_file(
        f"{PWM_PATH}/duty_cycle",
        duty_ns
    )

    actual = int(
        open(f"{PWM_PATH}/duty_cycle").read().strip()
    )

    print(
        f"requested={pulse_us:8.3f} us   "
        f"sysfs={actual / 1000:8.3f} us"
    )

def main():

    print()
    print("============================================================")
    print(" PWM OSCILLOSCOPE TEST")
    print("============================================================")
    print()
    print(f"PWM device : {PWM_PATH}")
    print(f"Period     : {PERIOD_US} us")
    print()

    setup()

    print("PWM enabled.")
    print()
    print("Connect oscilloscope:")
    print("  CH1 probe -> PWM signal")
    print("  CH1 ground -> Jetson GND")
    print()
    print("Starting at 1500 us.")
    print()

    try:

        set_pulse_us(1500)

        while True:

            print()
            print("Commands:")
            print("  1500       -> set 1500 us")
            print("  1500.5     -> set 1500.5 us")
            print("  1510       -> set 1510 us")
            print("  sweep      -> automatic sweep")
            print("  q          -> quit")
            print()

            command = input("> ").strip()

            if command.lower() == "q":
                break

            if command.lower() == "sweep":
                run_sweep()
                continue

            try:
                pulse = float(command)

                if pulse < 100:
                    print("Pulse too small.")
                    continue

                if pulse > 3000:
                    print("Pulse too large.")
                    continue

                set_pulse_us(pulse)

            except ValueError:
                print("Invalid command.")

    finally:

        print()
        print("Returning to 1500 us.")

        try:
            set_pulse_us(1500)
        except Exception:
            pass

        print("Leaving PWM enabled at 1500 us.")


def run_sweep():

    print()
    print("============================================================")
    print(" AUTOMATIC PWM SWEEP")
    print("============================================================")
    print()
    print("Each pulse width will remain active for 2 seconds.")
    print()
    print("Measure pulse width with the oscilloscope.")
    print()

    values = []

    # Fine sweep around the nominal center.
    for i in range(-20, 21):
        values.append(1500 + i * 0.5)

    for pulse in values:

        set_pulse_us(pulse)

        print(
            f"*** HOLD {pulse:.1f} us for 2 seconds ***"
        )

        time.sleep(2)

    print()
    print("Sweep complete.")
    print()

    set_pulse_us(1500)


if __name__ == "__main__":
    main()
