#!/usr/bin/env python3

import Jetson.GPIO as GPIO
import time
import sys
import termios
import tty
import select
import threading

# sudo busybox devmem 0x2430068 w 0x8
# sudo busybox devmem 0x2430070 w 0x8
# ============================================================
# USER CONFIGURATION
# ============================================================

# ---------------- GPIO pins ----------------
# Physical pin numbers on Jetson 40-pin header

STEP_PIN = 29
DIR_PIN  = 31
EN_PIN   = 33


# ------------------------------------------------------------
# LIMIT SWITCHES
# ------------------------------------------------------------
#
# IMPORTANT:
# These are physical Jetson GPIO pins connected to switches.
#
# Set these to your actual pins.
#
# Example:
#   LIMIT_MIN_PIN = 36
#   LIMIT_MAX_PIN = 37
#
# Set to None until you have connected the switches.
#
LIMIT_MIN_PIN = None
LIMIT_MAX_PIN = None

# Most mechanical switches should connect:
#
#       GPIO ---- switch ---- GND
#
# and use the Jetson's internal pull-up.
#
# Therefore:
LIMIT_ACTIVE_STATE = GPIO.LOW


# ============================================================
# CALIBRATION SETTINGS
# ============================================================

# Enable automatic calibration at startup.
#
# MUST be False if you do not have limit switches.
#
CALIBRATION_ENABLED = False

# Speed used while searching for an end.
# Units: steps/second
CALIBRATION_SPEED = 1000

# Speed used when approaching the terminal.
# Units: steps/second
CALIBRATION_CREEP_SPEED = 200

# Distance over which calibration begins slowing down.
# Units: steps
CALIBRATION_SLOWDOWN_STEPS = 500

# How quickly the motor accelerates/decelerates.
# Units: steps/second^2
CALIBRATION_ACCELERATION = 1000

# Maximum number of steps allowed while searching for
# a limit switch. This is a safety limit.
#
# You MUST set this large enough for your actuator.
#
CALIBRATION_MAX_TRAVEL = 200000


# ============================================================
# MANUAL CONTROL SETTINGS
# ============================================================

# Initial manual speed
MANUAL_START_SPEED = 10_000

# Minimum manual speed
MANUAL_MIN_SPEED = 50

# Maximum manual speed
MANUAL_MAX_SPEED = 50000

# Amount changed when pressing +/-
SPEED_INCREMENT = 100


# ============================================================
# MOTION SETTINGS
# ============================================================

# Acceleration for manual motion
MANUAL_ACCELERATION = 1500

# Maximum speed used by the max-speed test
#
# DO NOT immediately set this extremely high.
#
MAX_TEST_SPEED = 5000

# Acceleration used by max-speed test
MAX_TEST_ACCELERATION = 2000


# ============================================================
# TIMING
# ============================================================

# Keyboard polling interval
KEYBOARD_POLL_INTERVAL = 0.01

# Small delay after changing DIR before STEP
DIR_SETUP_TIME = 0.00001


# ============================================================
# GLOBAL STATE
# ============================================================

position = 0

travel_distance = None

current_speed = 0.0

running = True

emergency_stop = False

calibration_complete = False

manual_direction = 0

state_lock = threading.Lock()


# ============================================================
# GPIO SETUP
# ============================================================

GPIO.setmode(GPIO.BOARD)

GPIO.setup(
    STEP_PIN,
    GPIO.OUT,
    initial=GPIO.LOW
)

GPIO.setup(
    DIR_PIN,
    GPIO.OUT,
    initial=GPIO.LOW
)

GPIO.setup(
    EN_PIN,
    GPIO.OUT,
    initial=GPIO.HIGH
)


# Configure limit switches if present

if LIMIT_MIN_PIN is not None:

    GPIO.setup(
        LIMIT_MIN_PIN,
        GPIO.IN,
        pull_up_down=GPIO.PUD_UP
    )


if LIMIT_MAX_PIN is not None:

    GPIO.setup(
        LIMIT_MAX_PIN,
        GPIO.IN,
        pull_up_down=GPIO.PUD_UP
    )


# ============================================================
# TMC2209 CONTROL
# ============================================================

def enable_motor():

    # TMC2209 EN is active LOW

    GPIO.output(
        EN_PIN,
        GPIO.LOW
    )


def disable_motor():

    GPIO.output(
        EN_PIN,
        GPIO.HIGH
    )


def set_direction(direction):

    if direction > 0:

        GPIO.output(
            DIR_PIN,
            GPIO.HIGH
        )

    else:

        GPIO.output(
            DIR_PIN,
            GPIO.LOW
        )

    # Allow DIR signal to settle

    time.sleep(DIR_SETUP_TIME)


# ============================================================
# LIMIT SWITCH FUNCTIONS
# ============================================================

def min_limit_triggered():

    if LIMIT_MIN_PIN is None:

        return False

    return (
        GPIO.input(LIMIT_MIN_PIN)
        == LIMIT_ACTIVE_STATE
    )


def max_limit_triggered():

    if LIMIT_MAX_PIN is None:

        return False

    return (
        GPIO.input(LIMIT_MAX_PIN)
        == LIMIT_ACTIVE_STATE
    )


def any_limit_triggered():

    return (
        min_limit_triggered()
        or max_limit_triggered()
    )


# ============================================================
# KEYBOARD INPUT
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

        ready, _, _ = select.select(
            [sys.stdin],
            [],
            [],
            0
        )

        if ready:

            return sys.stdin.read(1)

        return None


# ============================================================
# LOW LEVEL STEP
# ============================================================

def step_once(delay):

    global position

    GPIO.output(
        STEP_PIN,
        GPIO.HIGH
    )

    time.sleep(delay)

    GPIO.output(
        STEP_PIN,
        GPIO.LOW
    )

    time.sleep(delay)

    with state_lock:

        position += (
            1 if GPIO.input(DIR_PIN)
            else -1
        )


# ============================================================
# MOVE AT SPEED
# ============================================================

def move_steps(
    steps,
    direction,
    speed
):

    if steps <= 0:

        return

    set_direction(direction)

    enable_motor()

    delay = 1.0 / (2.0 * speed)

    for _ in range(steps):

        if emergency_stop:

            break

        # Stop if appropriate limit reached

        if direction < 0 and min_limit_triggered():

            print("\nMIN limit reached.")

            break

        if direction > 0 and max_limit_triggered():

            print("\nMAX limit reached.")

            break

        step_once(delay)


# ============================================================
# ACCELERATED MOTION
# ============================================================

def move_with_acceleration(
    direction,
    target_speed,
    acceleration,
    duration=None,
    stop_condition=None
):

    """
    Move continuously while accelerating toward target_speed.

    If duration is supplied, stop after that amount of time.

    stop_condition may be a function returning True.
    """

    global current_speed

    set_direction(direction)

    enable_motor()

    start_time = time.monotonic()

    current_speed = 0.0

    while running and not emergency_stop:

        now = time.monotonic()

        elapsed = now - start_time

        if duration is not None:

            if elapsed >= duration:

                break

        # Acceleration

        current_speed += (
            acceleration
            * KEYBOARD_POLL_INTERVAL
        )

        if current_speed > target_speed:

            current_speed = target_speed

        # Check external stop condition

        if stop_condition is not None:

            if stop_condition():

                break

        if direction < 0 and min_limit_triggered():

            break

        if direction > 0 and max_limit_triggered():

            break

        if current_speed <= 0:

            continue

        delay = 1.0 / (
            2.0 * current_speed
        )

        step_once(delay)


    current_speed = 0


# ============================================================
# GRADUAL DECELERATION
# ============================================================

def decelerate_to_stop(
    direction,
    initial_speed,
    acceleration
):

    global current_speed

    speed = initial_speed

    while speed > 0:

        if emergency_stop:

            break

        delay = 1.0 / (
            2.0 * max(speed, 1)
        )

        step_once(delay)

        speed -= (
            acceleration
            * KEYBOARD_POLL_INTERVAL
        )

        if speed < 0:

            speed = 0

    current_speed = 0


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_toward_limit(
    direction,
    limit_function
):

    """
    Search for one terminal limit.

    Starts relatively fast and gradually slows
    near the end.

    Returns number of steps travelled.
    """

    print()

    if direction < 0:

        print(
            "Searching for MINIMUM limit..."
        )

    else:

        print(
            "Searching for MAXIMUM limit..."
        )

    set_direction(direction)

    enable_motor()

    steps = 0

    speed = 0.0

    while running and not emergency_stop:

        # Safety limit

        if steps >= CALIBRATION_MAX_TRAVEL:

            print(
                "\nERROR: calibration exceeded "
                "CALIBRATION_MAX_TRAVEL."
            )

            return None

        # Limit reached

        if limit_function():

            print(
                f"Limit detected after "
                f"{steps} steps."
            )

            return steps

        # Determine desired speed

        remaining_estimate = (
            CALIBRATION_SLOWDOWN_STEPS
        )

        # We don't know exact remaining distance
        # until the switch triggers.
        #
        # Therefore use a conservative
        # periodic slowdown near the end
        # based on an estimated step window.

        if (
            steps >
            CALIBRATION_MAX_TRAVEL
            - CALIBRATION_SLOWDOWN_STEPS
        ):

            desired_speed = (
                CALIBRATION_CREEP_SPEED
            )

        else:

            desired_speed = (
                CALIBRATION_SPEED
            )

        # Accelerate toward desired speed

        if speed < desired_speed:

            speed += (
                CALIBRATION_ACCELERATION
                * KEYBOARD_POLL_INTERVAL
            )

            speed = min(
                speed,
                desired_speed
            )

        delay = 1.0 / (
            2.0 * max(speed, 1)
        )

        step_once(delay)

        steps += 1

    return None


# ============================================================
# BETTER CALIBRATION WITH TWO-STAGE APPROACH
# ============================================================

def search_limit(direction, limit_function):

    """
    Search toward a limit.

    Phase 1:
        Fast movement.

    Phase 2:
        Once the configured search distance is
        reached, switch to creep speed.

    NOTE:
    Because there is no position information before
    hitting the switch, this is a conservative
    slowdown mechanism rather than distance-to-switch
    prediction.
    """

    set_direction(direction)

    enable_motor()

    steps = 0

    # First stage

    fast_speed = CALIBRATION_SPEED

    print(
        "  Searching at "
        f"{fast_speed:.0f} steps/s..."
    )

    while steps < CALIBRATION_MAX_TRAVEL:

        if emergency_stop:

            return None

        if limit_function():

            return steps

        # Once we're getting close to the safety
        # travel limit, slow down.

        if (
            steps >=
            CALIBRATION_MAX_TRAVEL
            - CALIBRATION_SLOWDOWN_STEPS
        ):

            break

        delay = 1.0 / (
            2.0 * fast_speed
        )

        step_once(delay)

        steps += 1

    # Slow final approach

    print(
        "  Switching to creep speed..."
    )

    creep_speed = CALIBRATION_CREEP_SPEED

    while steps < CALIBRATION_MAX_TRAVEL:

        if emergency_stop:

            return None

        if limit_function():

            return steps

        delay = 1.0 / (
            2.0 * creep_speed
        )

        step_once(delay)

        steps += 1

    print(
        "\nERROR: limit switch was not detected."
    )

    return None


def calibrate():

    global position
    global travel_distance
    global calibration_complete

    if (
        LIMIT_MIN_PIN is None
        or LIMIT_MAX_PIN is None
    ):

        print()
        print(
            "Calibration cannot run."
        )
        print(
            "Two limit switches are required."
        )

        return False

    print()
    print("=" * 60)
    print("STARTING CALIBRATION")
    print("=" * 60)

    enable_motor()

    # --------------------------------------------------------
    # STEP 1
    # Go toward MIN
    # --------------------------------------------------------

    print()
    print("1. Moving toward MIN limit...")

    min_steps = search_limit(
        -1,
        min_limit_triggered
    )

    if min_steps is None:

        disable_motor()

        return False

    # We define MIN as position zero

    position = 0

    time.sleep(0.5)

    # --------------------------------------------------------
    # STEP 2
    # Move toward MAX
    # --------------------------------------------------------

    print()
    print("2. Moving toward MAX limit...")

    max_steps = search_limit(
        +1,
        max_limit_triggered
    )

    if max_steps is None:

        disable_motor()

        return False

    travel_distance = (
        max_steps
    )

    print()
    print(
        f"Measured travel: "
        f"{travel_distance} steps"
    )

    # --------------------------------------------------------
    # STEP 3
    # Move to center
    # --------------------------------------------------------

    center_steps = (
        travel_distance // 2
    )

    print(
        f"3. Moving to center: "
        f"{center_steps} steps"
    )

    set_direction(-1)

    move_steps(
        center_steps,
        -1,
        CALIBRATION_CREEP_SPEED
    )

    position = (
        travel_distance
        - center_steps
    )

    calibration_complete = True

    print()
    print("=" * 60)
    print("CALIBRATION COMPLETE")
    print(
        f"Travel = {travel_distance} steps"
    )
    print(
        f"Center position = {position} steps"
    )
    print("=" * 60)

    return True


# ============================================================
# MANUAL MOTION
# ============================================================

def manual_move(direction, speed):

    """
    Move continuously at the requested speed.

    This is intentionally implemented as a blocking
    motion loop. Keyboard input is checked between
    short motion intervals.
    """

    global current_speed

    set_direction(direction)

    enable_motor()

    current_speed = 0.0

    while running and not emergency_stop:

        # Check limits

        if direction < 0:

            if min_limit_triggered():

                print(
                    "\nMIN limit reached."
                )

                break

        if direction > 0:

            if max_limit_triggered():

                print(
                    "\nMAX limit reached."
                )

                break

        # Accelerate

        if current_speed < speed:

            current_speed += (
                MANUAL_ACCELERATION
                * KEYBOARD_POLL_INTERVAL
            )

            current_speed = min(
                current_speed,
                speed
            )

        step_delay = 1.0 / (
            2.0 * max(
                current_speed,
                1
            )
        )

        step_once(step_delay)

        # Keyboard stop check

        key = keyboard.get_key()

        if key is not None:

            if key in (" ", "q"):

                break

    current_speed = 0


# ============================================================
# MAX SPEED TEST
# ============================================================

def max_velocity_test():

    global emergency_stop

    if (
        LIMIT_MIN_PIN is None
        or LIMIT_MAX_PIN is None
    ):

        print()
        print(
            "MAX SPEED TEST requires both "
            "limit switches."
        )

        return

    print()
    print("=" * 60)
    print("MAX VELOCITY TEST")
    print("=" * 60)

    print()
    print(
        "WARNING: actuator will travel "
        "from one end to the other."
    )

    print(
        f"Target speed: "
        f"{MAX_TEST_SPEED} steps/s"
    )

    print()
    print(
        "Press SPACE to abort."
    )

    time.sleep(2)

    if min_limit_triggered():

        direction = +1
        target_limit = max_limit_triggered

    elif max_limit_triggered():

        direction = -1
        target_limit = min_limit_triggered

    else:

        print(
            "Actuator is not currently at "
            "a terminal limit."
        )

        print(
            "Move it to an end before running "
            "the end-to-end speed test."
        )

        return

    set_direction(direction)

    enable_motor()

    speed = 0.0

    while not target_limit():

        if emergency_stop:

            return

        # Keyboard emergency stop

        key = keyboard.get_key()

        if key == " ":

            print(
                "\nEmergency stop."
            )

            return

        # Accelerate

        speed += (
            MAX_TEST_ACCELERATION
            * KEYBOARD_POLL_INTERVAL
        )

        if speed > MAX_TEST_SPEED:

            speed = MAX_TEST_SPEED

        delay = 1.0 / (
            2.0 * speed
        )

        step_once(delay)

    print()
    print(
        "Reached opposite terminal."
    )

    print(
        "Max velocity test complete."
    )


# ============================================================
# STATUS DISPLAY
# ============================================================

def print_status(speed):

    print()
    print("-" * 60)

    print(
        f"Position: "
        f"{position}"
    )

    if travel_distance is not None:

        percentage = (
            100.0
            * position
            / travel_distance
        )

        print(
            f"Travel: "
            f"{travel_distance} steps "
            f"({percentage:.1f}%)"
        )

    else:

        print(
            "Travel: not calibrated"
        )

    print(
        f"Speed: "
        f"{speed} steps/s"
    )

    print("-" * 60)


# ============================================================
# KEYBOARD CONTROLLER
# ============================================================

def keyboard_control():

    global running
    global emergency_stop
    global manual_direction
    global manual_speed

    manual_speed = MANUAL_START_SPEED

    print()
    print("=" * 60)
    print("KEYBOARD CONTROL")
    print("=" * 60)

    print()
    print("Controls:")
    print()
    print("  W / UP       Move forward")
    print("  S / DOWN     Move backward")
    print("  + / =        Increase speed")
    print("  -            Decrease speed")
    print("  SPACE        Stop")
    print("  C            Recalibrate")
    print("  T            Max velocity test")
    print("  P            Print status")
    print("  Q            Quit")
    print()

    print_status(
        manual_speed
    )

    while running:

        key = keyboard.get_key()

        if key is None:

            time.sleep(
                KEYBOARD_POLL_INTERVAL
            )

            continue

        key_lower = key.lower()

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if key_lower == "q":

            print(
                "\nQuitting..."
            )

            break

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        elif key == " ":

            print(
                "\nSTOP"
            )

            emergency_stop = True

            time.sleep(0.05)

            emergency_stop = False

        # ----------------------------------------------------
        # SPEED UP
        # ----------------------------------------------------

        elif key in ("+", "="):

            manual_speed += (
                SPEED_INCREMENT
            )

            manual_speed = min(
                manual_speed,
                MANUAL_MAX_SPEED
            )

            print_status(
                manual_speed
            )

        # ----------------------------------------------------
        # SPEED DOWN
        # ----------------------------------------------------

        elif key == "-":

            manual_speed -= (
                SPEED_INCREMENT
            )

            manual_speed = max(
                manual_speed,
                MANUAL_MIN_SPEED
            )

            print_status(
                manual_speed
            )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        elif key_lower == "p":

            print_status(
                manual_speed
            )

        # ----------------------------------------------------
        # RECALIBRATE
        # ----------------------------------------------------

        elif key_lower == "c":

            if (
                LIMIT_MIN_PIN is None
                or LIMIT_MAX_PIN is None
            ):

                print(
                    "\nCalibration unavailable: "
                    "two limit switches are required."
                )

            else:

                print("\nRecalibrating...")

                emergency_stop = False

                calibrate()

        # ----------------------------------------------------
        # MAX SPEED TEST
        # ----------------------------------------------------

        elif key_lower == "t":

            emergency_stop = False

            max_velocity_test()

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        elif key_lower == "w":

            print(
                f"\nMoving FORWARD "
                f"at {manual_speed} steps/s"
            )

            move_continuous_keyboard(
                +1,
                manual_speed
            )

        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        elif key_lower == "s":

            print(
                f"\nMoving BACKWARD "
                f"at {manual_speed} steps/s"
            )

            move_continuous_keyboard(
                -1,
                manual_speed
            )


# ============================================================
# KEYBOARD MOTION
# ============================================================

def move_continuous_keyboard(direction, speed):

    global emergency_stop
    global running

    set_direction(direction)
    enable_motor()

    current_speed_local = 0.0

    while running:

        # Read keyboard
        key = keyboard.get_key()

        if key is not None:

            # Space = stop current movement
            if key == " ":
                print("\nStopped.")
                break

            # Q = quit program
            if key.lower() == "q":
                running = False
                break

            # Increase speed
            if key in ("+", "="):

                speed += SPEED_INCREMENT
                speed = min(
                    speed,
                    MANUAL_MAX_SPEED
                )

                print(
                    f"\nSpeed: {speed} steps/s"
                )

            # Decrease speed
            elif key == "-":

                speed -= SPEED_INCREMENT
                speed = max(
                    speed,
                    MANUAL_MIN_SPEED
                )

                print(
                    f"\nSpeed: {speed} steps/s"
                )

        # Check MIN limit
        if direction < 0:

            if min_limit_triggered():

                print("\nMIN limit reached.")
                break

        # Check MAX limit
        if direction > 0:

            if max_limit_triggered():

                print("\nMAX limit reached.")
                break

        # Accelerate toward target speed
        if current_speed_local < speed:

            current_speed_local += (
                MANUAL_ACCELERATION
                * KEYBOARD_POLL_INTERVAL
            )

            current_speed_local = min(
                current_speed_local,
                speed
            )

        # Generate STEP pulse
        delay = 1.0 / (
            2.0 * max(
                current_speed_local,
                1
            )
        )

        step_once(delay)

    disable_motor()

# ============================================================
# MAIN
# ============================================================

keyboard = Keyboard()


try:

    keyboard.start()

    enable_motor()

    print()
    print("=" * 60)
    print("FUYU ACTUATOR CONTROLLER")
    print("=" * 60)

    print()

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    if CALIBRATION_ENABLED:

        success = calibrate()

        if not success:

            print()
            print(
                "Calibration failed."
            )

            print(
                "Motor disabled."
            )

            disable_motor()

            sys.exit(1)

    else:

        print(
            "Automatic calibration is DISABLED."
        )

        print(
            "Position is not known."
        )

        print()

    # --------------------------------------------------------
    # KEYBOARD MODE
    # --------------------------------------------------------

    keyboard_control()


except KeyboardInterrupt:

    print(
        "\nInterrupted."
    )


finally:

    print(
        "\nDisabling motor..."
    )

    disable_motor()

    keyboard.stop()

    GPIO.output(
        STEP_PIN,
        GPIO.LOW
    )

    GPIO.cleanup()

    print(
        "GPIO cleaned up."
    )