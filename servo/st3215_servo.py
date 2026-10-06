#!/usr/bin/env python3
"""
Keyboard control for an ST3215 servo (Waveshare driver board -> Jetson
Orin Nano), built on the `st3215` pip library:
    https://github.com/Mickael-Roger/python-st3215

Install:
    pip3 install st3215

Board setup (one-time per boot unless you've flashed SERIAL_FORWARDING=true):
    On the driver board's screen, select "Start Serial Forwarding" before
    running this script. While forwarding, the USB link runs at 115200
    baud (not the servo's own 1,000,000 baud - the board handles that
    internally).

Controls:
  a / LEFT     : jog down by one step
  d / RIGHT    : jog up by one step
  [ / ]        : decrease / increase jog step size
  c            : set CURRENT position as the new middle (2048) - this
                 calls DefineMiddle(), which writes a firmware position
                 correction so this physical spot is permanently 2048,
                 even after power-off, until you redefine it again.
  m            : move to middle (2048)
  - / =        : decrease / increase the sweep limit, in degrees from middle
  s            : start/stop continuous sweep between (middle - limit) and
                 (middle + limit)
  SPACE        : stop sweep (if running) and hold current position
  t            : toggle torque on/off
  i            : print diagnostics (position, load, voltage, current)
  q / ESC      : quit
"""

import sys
import time
import tty
import termios
import select

from st3215 import ST3215

# ------------------------- Config -------------------------------------
PORT = "/dev/ttyUSB0"     # change if needed: ls /dev/ttyUSB* /dev/ttyACM*
BAUD = 115200               # driver board's Serial Forwarding mode baud rate
SERVO_ID = 1
STEP = 40                   # jog step, in servo position units (4096 = 360deg)
SPEED = 1500
ACC = 50

STEPS_PER_DEG = 4096 / 360.0
MIDDLE = 2048                # logical middle; becomes accurate after 'c'
LIMIT_DEG = 45                # default +/- sweep range in degrees
SWEEP_DWELL = 0.3             # seconds to pause at each end before reversing


def connect(port, baud):
    """
    The st3215 library's ST3215() constructor only takes `device` (no baud
    argument) and opens the port at 1,000,000 baud internally via a
    Dynamixel/Feetech-SDK-style portHandler. To talk to it at 115200 (the
    Waveshare driver board's Serial Forwarding mode baud rate), we reach
    into portHandler.ser - the underlying pyserial object - and change its
    baud rate directly, which takes effect immediately on an already-open
    port.
    """
    servo = ST3215(port)
    try:
        servo.portHandler.ser.baudrate = baud
    except AttributeError:
        print(f"Warning: could not set baud rate on portHandler.ser - "
              f"inspect it with: python3 -c \"from st3215 import ST3215; "
              f"s = ST3215('{port}'); print(dir(s.portHandler))\"")
    return servo


# ------------------------- Keyboard input --------------------------------
def get_key(timeout=0.1):
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    if not dr:
        return None
    ch = sys.stdin.read(1)
    if ch == '\x1b':
        dr2, _, _ = select.select([sys.stdin], [], [], 0.01)
        if dr2:
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                dr3, _, _ = select.select([sys.stdin], [], [], 0.01)
                if dr3:
                    ch3 = sys.stdin.read(1)
                    return {'C': 'RIGHT', 'D': 'LEFT'}.get(ch3)
        return 'ESC'
    return ch


def deg_to_steps(deg):
    return int(round(deg * STEPS_PER_DEG))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def sweep_step(servo, low, high, going_up):
    """One leg of the sweep: move to the target, wait until arrival or an
    abort key, then return (new going_up, aborted_key)."""
    target = high if going_up else low
    servo.MoveTo(SERVO_ID, target, speed=SPEED, acc=ACC, wait=False)
    while True:
        key = get_key(0.05)
        if key is not None:
            return going_up, key
        moving = servo.IsMoving(SERVO_ID)
        if not moving:
            break
    time.sleep(SWEEP_DWELL)
    return (not going_up), None


def print_diagnostics(servo, low, high):
    pos = servo.ReadPosition(SERVO_ID)
    load = servo.ReadLoad(SERVO_ID)
    volt = servo.ReadVoltage(SERVO_ID)
    curr = servo.ReadCurrent(SERVO_ID)
    print(f"  position={pos}  middle={MIDDLE}  sweep_range=[{low},{high}] "
          f"(+/-{LIMIT_DEG} deg)  load={load}  voltage={volt}V  current={curr}mA")


def main():
    global STEP, LIMIT_DEG, MIDDLE

    servo = connect(PORT, BAUD)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    torque_on = True
    sweeping = False
    going_up = True

    try:
        tty.setcbreak(fd)
        servo.StartServo(SERVO_ID)

        pos = servo.ReadPosition(SERVO_ID)
        if pos is None:
            print("Warning: could not read initial position - check wiring/ID/baud.")
            pos = MIDDLE
        print(f"Connected on {PORT}. Starting position: {pos}")
        print("Controls: a/LEFT=down  d/RIGHT=up  [/]=step size  "
              "c=set middle here  m=go to middle")
        print("          -/+=sweep range  s=start/stop sweep  SPACE=hold  "
              "t=torque  i=diagnostics  q/ESC=quit")
        print(f"Step size: {STEP}   Sweep range: +/-{LIMIT_DEG} deg")

        while True:
            limit_steps = deg_to_steps(LIMIT_DEG)
            low = clamp(MIDDLE - limit_steps, 0, 4095)
            high = clamp(MIDDLE + limit_steps, 0, 4095)

            if sweeping:
                going_up, abort_key = sweep_step(servo, low, high, going_up)
                if abort_key is None:
                    continue
                key = abort_key
                sweeping = False
                print("sweep stopped")
            else:
                key = get_key(0.1)
                if key is None:
                    continue

            if key in ('q', 'ESC'):
                break
            elif key in ('a', 'LEFT'):
                pos = clamp(pos - STEP, 0, 4095)
                servo.MoveTo(SERVO_ID, pos, speed=SPEED, acc=ACC, wait=False)
                print(f"pos -> {pos}")
            elif key in ('d', 'RIGHT'):
                pos = clamp(pos + STEP, 0, 4095)
                servo.MoveTo(SERVO_ID, pos, speed=SPEED, acc=ACC, wait=False)
                print(f"pos -> {pos}")
            elif key == '[':
                STEP = max(1, STEP - 10)
                print(f"step size -> {STEP}")
            elif key == ']':
                STEP = min(500, STEP + 10)
                print(f"step size -> {STEP}")
            elif key == 'c':
                servo.DefineMiddle(SERVO_ID)
                MIDDLE = 2048
                pos = MIDDLE
                print(f"Middle redefined at this physical position (now reports as {MIDDLE}).")
            elif key == 'm':
                pos = MIDDLE
                servo.MoveTo(SERVO_ID, pos, speed=SPEED, acc=ACC, wait=False)
                print(f"moving to middle -> {pos}")
            elif key == '-':
                LIMIT_DEG = max(1, LIMIT_DEG - 5)
                print(f"sweep range -> +/-{LIMIT_DEG} deg")
            elif key == '=':
                LIMIT_DEG = min(180, LIMIT_DEG + 5)
                print(f"sweep range -> +/-{LIMIT_DEG} deg")
            elif key == 's':
                sweeping = True
                going_up = True
                print(f"sweep started: {low} <-> {high} (+/-{LIMIT_DEG} deg around {MIDDLE})")
            elif key == ' ':
                actual = servo.ReadPosition(SERVO_ID)
                if actual is not None:
                    pos = actual
                servo.MoveTo(SERVO_ID, pos, speed=SPEED, acc=ACC, wait=False)
                print(f"holding at {pos}")
            elif key == 't':
                torque_on = not torque_on
                if torque_on:
                    servo.StartServo(SERVO_ID)
                else:
                    servo.StopServo(SERVO_ID)
                print(f"torque {'ON' if torque_on else 'OFF'}")
            elif key == 'i':
                print_diagnostics(servo, low, high)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        try:
            servo.close()
        except Exception:
            pass
        print("\nExiting, connection closed.")


if __name__ == "__main__":
    main()