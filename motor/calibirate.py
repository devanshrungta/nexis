#!/usr/bin/env python3

import time
import serial
import Jetson.GPIO as GPIO

# ============================================================
# JETSON PINS
# ============================================================

STEP_PIN = 29
DIR_PIN  = 31
EN_PIN   = 33
DIAG_PIN = 37

# Change this after checking:
# ls -l /dev/ttyTHS*
UART_PORT = "/dev/ttyTHS0"
UART_BAUD = 115200

# ============================================================
# MOTOR SETTINGS
# ============================================================

# Start SLOW for calibration.
STEP_DELAY = 0.0025       # 400 steps/sec approximately

# Number of steps used as the maximum possible travel.
# This is intentionally larger than the expected travel.
MAX_CAL_STEPS = 300000

# Number of steps to move away from the detected end
# before beginning the opposite direction calibration.
BACKOFF_STEPS = 500

# Steps to move after finding the second end.
CENTER_EXTRA_DELAY = 0.5

# TMC2209 UART address
TMC_ADDRESS = 0

# StallGuard threshold.
#
# This MUST be tuned experimentally.
# Start around 60 and adjust.
SGTHRS = 60

# ============================================================
# TMC2209 REGISTERS
# ============================================================

REG_GCONF      = 0x00
REG_GSTAT      = 0x01
REG_IHOLD_IRUN = 0x10
REG_TPOWERDOWN = 0x11
REG_TPWMTHRS   = 0x13
REG_TCOOLTHRS  = 0x14
REG_SGTHRS     = 0x40
REG_SG_RESULT  = 0x41
REG_CHOPCONF   = 0x6C
REG_PWMCONF    = 0x70


# ============================================================
# CRC
# ============================================================

def crc8(data):

    crc = 0

    for byte in data:

        for _ in range(8):

            if (crc >> 7) & 1:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

            byte >>= 1

    return crc


# ============================================================
# TMC2209 UART
# ============================================================

class TMC2209:

    def __init__(self, port, baud=115200, address=0):

        self.address = address

        self.ser = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05
        )

        time.sleep(0.1)

    def write_reg(self, reg, value):

        packet = bytearray([
            0x05,
            self.address,
            reg | 0x80,
            (value >> 24) & 0xFF,
            (value >> 16) & 0xFF,
            (value >> 8) & 0xFF,
            value & 0xFF
        ])

        packet.append(crc8(packet))

        self.ser.write(packet)
        self.ser.flush()

        time.sleep(0.001)

    def read_reg(self, reg):

        request = bytearray([
            0x05,
            self.address,
            reg & 0x7F
        ])

        request.append(crc8(request))

        self.ser.reset_input_buffer()

        self.ser.write(request)
        self.ser.flush()

        response = self.ser.read(8)

        if len(response) != 8:
            return None

        if crc8(response[:7]) != response[7]:
            return None

        value = (
            (response[3] << 24) |
            (response[4] << 16) |
            (response[5] << 8) |
            response[6]
        )

        return value

    def setup_stallguard(self):

        print("Configuring TMC2209...")

        # ----------------------------------------------------
        # GCONF
        #
        # Enable SpreadCycle.
        # StallGuard works with SpreadCycle.
        # ----------------------------------------------------

        gconf = self.read_reg(REG_GCONF)

        if gconf is None:
            print("ERROR: Cannot communicate with TMC2209 over UART.")
            return False

        print("GCONF =", hex(gconf))

        # en_spreadcycle = bit 2
        gconf |= (1 << 2)

        self.write_reg(REG_GCONF, gconf)

        # ----------------------------------------------------
        # Current
        #
        # IRUN = 16
        # IHOLD = 8
        #
        # Start conservatively.
        # ----------------------------------------------------

        ihold = 8
        irun = 16

        ihold_irun = (
            ihold |
            (irun << 8)
        )

        self.write_reg(
            REG_IHOLD_IRUN,
            ihold_irun
        )

        # ----------------------------------------------------
        # TPOWERDOWN
        # ----------------------------------------------------

        self.write_reg(
            REG_TPOWERDOWN,
            20
        )

        # ----------------------------------------------------
        # TCOOLTHRS
        #
        # Enables StallGuard over the movement speed range.
        # ----------------------------------------------------

        self.write_reg(
            REG_TCOOLTHRS,
            0xFFFFF
        )

        # ----------------------------------------------------
        # TPWMTHRS = 0
        #
        # Keeps operation in SpreadCycle.
        # ----------------------------------------------------

        self.write_reg(
            REG_TPWMTHRS,
            0
        )

        # ----------------------------------------------------
        # StallGuard threshold
        # ----------------------------------------------------

        self.write_reg(
            REG_SGTHRS,
            SGTHRS
        )

        print("StallGuard threshold =", SGTHRS)

        return True

    def sg_result(self):

        value = self.read_reg(REG_SG_RESULT)

        if value is None:
            return None

        return value & 0x3FF

    def close(self):

        self.ser.close()


# ============================================================
# GPIO
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

GPIO.setup(
    DIAG_PIN,
    GPIO.IN
)


# ============================================================
# MOTOR CONTROL
# ============================================================

def enable_motor():

    GPIO.output(EN_PIN, GPIO.LOW)
    time.sleep(0.1)


def disable_motor():

    GPIO.output(EN_PIN, GPIO.HIGH)


def step_once():

    GPIO.output(STEP_PIN, GPIO.HIGH)

    time.sleep(STEP_DELAY)

    GPIO.output(STEP_PIN, GPIO.LOW)

    time.sleep(STEP_DELAY)


def move_steps(steps, direction):

    GPIO.output(
        DIR_PIN,
        GPIO.HIGH if direction > 0 else GPIO.LOW
    )

    for _ in range(steps):

        step_once()


# ============================================================
# SENSORLESS STALL DETECTION
# ============================================================

def stall_detected(tmc):

    # DIAG is the primary stall signal.
    diag = GPIO.input(DIAG_PIN)

    if diag:
        return True

    # SG_RESULT gives us a second indication.
    sg = tmc.sg_result()

    if sg is None:
        return False

    # Very low SG_RESULT means the motor is heavily loaded.
    #
    # This value is intentionally conservative.
    # It may need adjustment for your actuator.
    if sg < 30:
        return True

    return False


# ============================================================
# CALIBRATE ONE END
# ============================================================

def find_end(tmc, direction):

    print()
    print(
        "Searching for",
        "FORWARD" if direction > 0 else "BACKWARD",
        "end..."
    )

    GPIO.output(
        DIR_PIN,
        GPIO.HIGH if direction > 0 else GPIO.LOW
    )

    time.sleep(0.1)

    for step in range(MAX_CAL_STEPS):

        step_once()

        if step > 100:

            if stall_detected(tmc):

                print(
                    "STALL DETECTED after",
                    step,
                    "steps"
                )

                # Stop immediately
                time.sleep(0.1)

                return step

    print("WARNING: Maximum calibration distance reached.")

    return None


# ============================================================
# BACKOFF
# ============================================================

def backoff(direction):

    print(
        "Backing off",
        BACKOFF_STEPS,
        "steps..."
    )

    move_steps(
        BACKOFF_STEPS,
        -direction
    )

    time.sleep(0.2)


# ============================================================
# MAIN CALIBRATION
# ============================================================

def calibrate(tmc):

    print()
    print("====================================")
    print(" SENSORLESS ACTUATOR CALIBRATION")
    print("====================================")
    print()

    enable_motor()

    # --------------------------------------------------------
    # Find first end
    # --------------------------------------------------------

    end1 = find_end(
        tmc,
        +1
    )

    if end1 is None:

        print("First end was not detected.")
        return False

    backoff(+1)

    # --------------------------------------------------------
    # Find second end
    # --------------------------------------------------------

    end2 = find_end(
        tmc,
        -1
    )

    if end2 is None:

        print("Second end was not detected.")
        return False

    print()
    print("====================================")
    print("CALIBRATION COMPLETE")
    print("====================================")

    print("End 1 steps:", end1)
    print("End 2 steps:", end2)

    travel = end2 + BACKOFF_STEPS

    print("Estimated travel:", travel, "steps")

    # --------------------------------------------------------
    # Move to center
    # --------------------------------------------------------

    center = travel // 2

    print()
    print("Moving to center:", center, "steps")

    move_steps(
        center,
        -1
    )

    time.sleep(CENTER_EXTRA_DELAY)

    print()
    print("CENTER POSITION REACHED")

    return True


# ============================================================
# PROGRAM
# ============================================================

tmc = None

try:

    print()
    print("FUYU SENSORLESS CALIBRATION")
    print()

    tmc = TMC2209(
        UART_PORT,
        UART_BAUD,
        TMC_ADDRESS
    )

    if not tmc.setup_stallguard():

        raise RuntimeError(
            "TMC2209 UART communication failed."
        )

    calibrate(tmc)

finally:

    disable_motor()

    if tmc is not None:
        tmc.close()

    GPIO.cleanup()

    print()
    print("Calibration program finished.")