import Jetson.GPIO as GPIO
import time
import curses


# ============================================================
# Configuration
# ============================================================

SERVO_PIN = 32

HOME_ANGLE = 90       # Starting position
LIMIT = 90            # +/- 30 degrees from home

STEP = 1              # Degrees per key press

RETURN_ANGLE = 0

PWM_FREQUENCY = 50


# ============================================================
# Limits
# ============================================================

MIN_ANGLE = max(0, HOME_ANGLE - LIMIT)
MAX_ANGLE = min(180, HOME_ANGLE + LIMIT)


# ============================================================
# GPIO
# ============================================================

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, PWM_FREQUENCY)
pwm.start(0)


# ============================================================
# Servo functions
# ============================================================

def angle_to_duty(angle):

    angle = max(0, min(180, angle))

    return 2.5 + (angle / 180.0) * 10.0


def set_angle(angle):

    duty = angle_to_duty(angle)

    pwm.ChangeDutyCycle(duty)

    time.sleep(0.1)

    # Stop PWM to reduce jitter
    pwm.ChangeDutyCycle(0)


# ============================================================
# Keyboard interface
# ============================================================

def main(stdscr):

    current_angle = HOME_ANGLE

    # Don't wait indefinitely for key
    stdscr.nodelay(True)

    # Hide cursor
    curses.curs_set(0)

    stdscr.clear()

    # Move to home
    set_angle(current_angle)

    while True:

        stdscr.clear()

        stdscr.addstr(
            0,
            0,
            "========================================"
        )

        stdscr.addstr(
            1,
            0,
            "       Keyboard Servo Controller"
        )

        stdscr.addstr(
            2,
            0,
            "========================================"
        )

        stdscr.addstr(
            4,
            0,
            f"Current angle : {current_angle} deg"
        )

        stdscr.addstr(
            5,
            0,
            f"Allowed range : {MIN_ANGLE} - {MAX_ANGLE} deg"
        )

        stdscr.addstr(
            7,
            0,
            "Controls:"
        )

        stdscr.addstr(
            8,
            0,
            "  UP / RIGHT    Increase"
        )

        stdscr.addstr(
            9,
            0,
            "  DOWN / LEFT   Decrease"
        )

        stdscr.addstr(
            10,
            0,
            "  Q             Quit"
        )

        stdscr.addstr(
            12,
            0,
            "Press a key..."
        )

        stdscr.refresh()

        # Get key
        key = stdscr.getch()

        # ----------------------------------------------------
        # Increase
        # ----------------------------------------------------

        if key == curses.KEY_UP or key == curses.KEY_RIGHT:

            new_angle = current_angle + STEP

            if new_angle <= MAX_ANGLE:

                current_angle = new_angle

                set_angle(current_angle)

        # ----------------------------------------------------
        # Decrease
        # ----------------------------------------------------

        elif key == curses.KEY_DOWN or key == curses.KEY_LEFT:

            new_angle = current_angle - STEP

            if new_angle >= MIN_ANGLE:

                current_angle = new_angle

                set_angle(current_angle)

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        elif key == ord("q") or key == ord("Q"):

            break

        time.sleep(0.01)


# ============================================================
# Run
# ============================================================

try:

    curses.wrapper(main)

finally:

    print("Returning servo to 0 degrees...")

    set_angle(RETURN_ANGLE)

    time.sleep(0.5)

    pwm.stop()
    GPIO.cleanup()

    print("Servo returned to 0 degrees")
    print("GPIO cleaned up")
