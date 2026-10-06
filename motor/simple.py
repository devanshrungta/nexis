import Jetson.GPIO as GPIO
import time

STEP_PIN = 32
DIR_PIN = 33

GPIO.setmode(GPIO.BOARD)

GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)


def move(steps, direction, step_delay=0.002):
    GPIO.output(DIR_PIN, GPIO.HIGH if direction else GPIO.LOW)

    for _ in range(steps):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(step_delay)

        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(step_delay)


try:
    print("Forward")
    move(500, True)

    time.sleep(1)

    print("Reverse")
    move(500, False)

finally:
    GPIO.output(STEP_PIN, GPIO.LOW)
    GPIO.cleanup()
