import serial
import time

PORT = "/dev/ttyTHS1"
BAUD = 115200
ADDRESS = 0


def crc8(data):
    crc = 0

    for byte in data:
        for _ in range(8):
            if ((crc >> 7) ^ (byte & 1)):
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

            byte >>= 1

    return crc


ser = serial.Serial(
    PORT,
    BAUD,
    bytesize=8,
    parity=serial.PARITY_NONE,
    stopbits=1,
    timeout=0.5
)

time.sleep(0.2)

# Read GCONF (register 0x00)
packet = bytearray([
    0x05,       # sync
    ADDRESS,    # slave address
    0x80 | 0x00 # read GCONF
])

packet.append(crc8(packet))

print("Sending:", packet.hex())

ser.reset_input_buffer()

ser.write(packet)
ser.flush()

time.sleep(0.1)

data = ser.read(12)

print("Received:", data.hex())
print("Bytes received:", len(data))

ser.close()