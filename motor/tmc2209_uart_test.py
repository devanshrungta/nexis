#!/usr/bin/env python3

import serial
import time


PORT = "/dev/ttyTHS1"
BAUD = 115200
ADDRESS = 0


def crc8(data):
    crc = 0

    for byte in data:
        for _ in range(8):
            if ((crc >> 7) ^ (byte & 0x01)):
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

            byte >>= 1

    return crc


def main():

    print(f"Opening {PORT} at {BAUD}...")

    ser = serial.Serial(
        port=PORT,
        baudrate=BAUD,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=1.0,
    )

    time.sleep(0.5)

    # TMC2209 READ GCONF (register 0x00)
    packet = bytes([
        0x05,            # sync
        ADDRESS,         # slave address
        0x00,            # register address, read
    ])

    packet += bytes([crc8(packet)])

    print("TX:", packet.hex(" "))

    ser.reset_input_buffer()

    ser.write(packet)
    ser.flush()

    # Give the driver time to switch from RX to TX.
    time.sleep(0.01)

    data = ser.read(32)

    print("RX:", data.hex(" "))
    print("RX bytes:", len(data))

    if not data:
        print("\nNO DATA")
        print("The UART line is not receiving anything.")

    elif len(data) == 2 and data == b"\x00\x00":
        print("\nONLY 00 00 RECEIVED")
        print("This strongly suggests the TMC response is not reaching the Jetson.")

    else:
        print("\nRaw response received.")

        # Look for a valid TMC response:
        # 05 FF 00 DATA[4] CRC
        for i in range(max(0, len(data) - 7)):

            frame = data[i:i + 8]

            if len(frame) != 8:
                continue

            if frame[0] != 0x05:
                continue

            if frame[1] != 0xFF:
                continue

            if frame[2] != 0x00:
                continue

            if crc8(frame[:7]) != frame[7]:
                continue

            print("\nVALID TMC2209 RESPONSE FOUND")

            print("Frame:", frame.hex(" "))

            gconf = int.from_bytes(
                frame[3:7],
                byteorder="big"
            )

            print(f"GCONF = 0x{gconf:08X}")

            break

    ser.close()


if __name__ == "__main__":
    main()