#!/usr/bin/env python3

import os
import sys
import time
import select
import termios
import tty


# ============================================================
# Configuration
# ============================================================

PWM_CHIP = "/sys/class/pwm/pwmchip3"
PWM_PATH = "/sys/class/pwm/pwmchip3/pwm0"

PWM_CHANNEL = 0

# Servo pulse limits
MIN_PULSE_US = 500
MAX_PULSE_US = 2500

# Neutral position
NEUTRAL_PULSE_US = 1500

# Servo PWM frequency
PWM_FREQUENCY = 50

# PWM period
PERIOD_NS = int(1_000_000_000 / PWM_FREQUENCY)

# Manual movement speed
#
# Hold an arrow key:
#   500 us/s means 1000 us takes 2 seconds
#
MANUAL_SPEED_US_PER_SEC = 500.0

# Oscillation speed
#
# 500 -> 2500 takes ~4 seconds
#
OSCILLATION_SPEED_US_PER_SEC = 500.0


# ============================================================
# PWM functions
# ============================================================

def write_sysfs(path, value):
    with open(path, "w") as f:
        f.write(str(value))


def set_pulse_us(pulse_us):
    """Set servo pulse width in microseconds."""

    pulse_us = max(
        MIN_PULSE_US,
        min(MAX_PULSE_US, pulse_us)
    )

    pulse_ns = int(round(pulse_us * 1000))

    write_sysfs(
        f"{PWM_PATH}/duty_cycle",
        pulse_ns
    )


def setup_pwm():

    # Export PWM0 if it isn't already exported
    if not os.path.exists(PWM_PATH):

        write_sysfs(
            f"{PWM_CHIP}/export",
            PWM_CHANNEL
        )

        time.sleep(0.1)

    # Disable before changing configuration
    try:
        write_sysfs(
            f"{PWM_PATH}/enable",
            0
        )
    except Exception:
        pass

    # 50 Hz
    write_sysfs(
        f"{PWM_PATH}/period",
        PERIOD_NS
    )

    # Start at neutral
    set_pulse_us(NEUTRAL_PULSE_US)

    # Enable hardware PWM
    write_sysfs(
        f"{PWM_PATH}/enable",
        1
    )


def stop_pwm():

    try:
        write_sysfs(
            f"{PWM_PATH}/enable",
            0
        )
    except Exception:
        pass


# ============================================================
# Keyboard helper
# ============================================================

class Keyboard:

    def __init__(self):

        self.fd = sys.stdin.fileno()

        self.old_settings = termios.tcgetattr(
            self.fd
        )

    def start(self):

        tty.setcbreak(self.fd)

    def stop(self):

        termios.tcsetattr(
            self.fd,
            termios.TCSADRAIN,
            self.old_settings
        )

    def get_key(self):

        """
        Non-blocking key read.

        Returns:
            character/string if a key is available
            None otherwise
        """

        readable, _, _ = select.select(
            [self.fd],
            [],
            [],
            0
        )

        if not readable:
            return None

        return os.read(
            self.fd,
            1
        ).decode(
            errors="ignore"
        )


# ============================================================
# Read arrow key
# ============================================================

def read_escape_sequence(keyboard):

    """
    Handle terminal escape sequences.

    Arrow keys normally arrive as:

        LEFT  = ESC [ D
        RIGHT = ESC [ C
    """

    if key := keyboard.get_key():

        if key == "[":

            if key2 := keyboard.get_key():

                if key2 == "D":
                    return "LEFT"

                if key2 == "C":
                    return "RIGHT"

    return None


# ============================================================
# Manual mode
# ============================================================

def manual_mode(keyboard):

    pulse = float(NEUTRAL_PULSE_US)

    direction = 0

    last_time = time.monotonic()

    print()
    print("==============================================")
    print("MANUAL MODE")
    print("==============================================")
    print()
    print("Hold LEFT  : decrease pulse width")
    print("Hold RIGHT : increase pulse width")
    print()
    print("A : decrease")
    print("D : increase")
    print()
    print("SPACE : stop movement")
    print("N     : neutral (1500 us)")
    print("O     : oscillation mode")
    print("Q     : quit")
    print()
    print(
        f"Range: {MIN_PULSE_US} - "
        f"{MAX_PULSE_US} us"
    )
    print()
    print(f"Pulse: {pulse:.1f} us")

    while True:

        now = time.monotonic()

        dt = now - last_time

        last_time = now

        # ----------------------------------------------------
        # Process keyboard input
        # ----------------------------------------------------

        key = keyboard.get_key()

        if key is not None:

            # --------------------------------------------
            # Escape sequence
            # --------------------------------------------

            if key == "\x1b":

                # Wait very briefly for '['
                time.sleep(0.001)

                sequence = keyboard.get_key()

                if sequence == "[":

                    time.sleep(0.001)

                    arrow = keyboard.get_key()

                    if arrow == "D":

                        direction = -1

                    elif arrow == "C":

                        direction = 1

                continue

            # --------------------------------------------
            # Normal keys
            # --------------------------------------------

            key = key.lower()

            # Quit
            if key == "q":

                return "quit"

            # Oscillation
            if key == "o":

                return "oscillate"

            # Neutral
            if key == "n":

                pulse = float(
                    NEUTRAL_PULSE_US
                )

                set_pulse_us(pulse)

                print(
                    f"\rPulse: {pulse:7.1f} us",
                    end="",
                    flush=True
                )

            # Stop
            elif key == " ":

                direction = 0

            # Left
            elif key in ("a", "j"):

                direction = -1

            # Right
            elif key in ("d", "l"):

                direction = 1

        # ----------------------------------------------------
        # Continuous movement
        # ----------------------------------------------------

        if direction != 0:

            pulse += (
                direction
                * MANUAL_SPEED_US_PER_SEC
                * dt
            )

            # Clamp at limits
            if pulse <= MIN_PULSE_US:

                pulse = float(MIN_PULSE_US)

            elif pulse >= MAX_PULSE_US:

                pulse = float(MAX_PULSE_US)

            set_pulse_us(pulse)

            print(
                f"\rPulse: {pulse:7.1f} us",
                end="",
                flush=True
            )

        time.sleep(0.005)


# ============================================================
# Oscillation mode
# ============================================================

def oscillation_mode(keyboard):

    pulse = float(MIN_PULSE_US)

    direction = 1

    last_time = time.monotonic()

    print()
    print("==============================================")
    print("OSCILLATION MODE")
    print("==============================================")
    print()
    print(
        f"{MIN_PULSE_US} -> "
        f"{MAX_PULSE_US} -> "
        f"{MIN_PULSE_US} us"
    )
    print()
    print(
        f"Speed: "
        f"{OSCILLATION_SPEED_US_PER_SEC} us/s"
    )
    print()
    print("Press Q to stop oscillation")
    print()

    while True:

        now = time.monotonic()

        dt = now - last_time

        last_time = now

        # ----------------------------------------------------
        # Check keyboard
        # ----------------------------------------------------

        key = keyboard.get_key()

        if key is not None:

            if key.lower() == "q":

                return

            # Any other key is ignored

        # ----------------------------------------------------
        # Move continuously
        # ----------------------------------------------------

        pulse += (
            direction
            * OSCILLATION_SPEED_US_PER_SEC
            * dt
        )

        # Upper limit
        if pulse >= MAX_PULSE_US:

            pulse = float(MAX_PULSE_US)

            direction = -1

        # Lower limit
        elif pulse <= MIN_PULSE_US:

            pulse = float(MIN_PULSE_US)

            direction = 1

        set_pulse_us(pulse)

        print(
            f"\rPulse: {pulse:7.1f} us",
            end="",
            flush=True
        )

        time.sleep(0.005)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("==============================================")
    print("       DS3240 SERVO CONTROLLER")
    print("==============================================")
    print()
    print(f"PWM device : {PWM_PATH}")
    print(f"Frequency  : {PWM_FREQUENCY} Hz")
    print(f"Period     : {PERIOD_NS} ns")
    print(f"Pulse range: {MIN_PULSE_US} - {MAX_PULSE_US} us")
    print(f"Neutral    : {NEUTRAL_PULSE_US} us")
    print()

    setup_pwm()

    keyboard = Keyboard()

    try:

        keyboard.start()

        mode = "manual"

        while True:

            if mode == "manual":

                result = manual_mode(keyboard)

                if result == "quit":
                    break

                if result == "oscillate":
                    mode = "oscillate"

            elif mode == "oscillate":

                oscillation_mode(keyboard)

                # Return to manual after oscillation
                mode = "manual"

    except KeyboardInterrupt:

        pass

    finally:

        # Restore terminal BEFORE printing normally
        keyboard.stop()

        print()
        print()
        print("Returning servo to neutral...")
        
        try:

            set_pulse_us(
                NEUTRAL_PULSE_US
            )

            time.sleep(0.5)

        except Exception as e:

            print(
                f"Could not return to neutral: {e}"
            )

        stop_pwm()

        print("PWM stopped.")
        print("Done.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
