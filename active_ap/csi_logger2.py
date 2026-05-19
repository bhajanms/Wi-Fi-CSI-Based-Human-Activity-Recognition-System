import serial
import time

PORT = "COM10"
BAUD = 115200
OUT = "fall101.csv"

DURATION = 60    # 1 minutes
MIN_PACKETS = 200     # minimum expected packets

ser = serial.Serial(PORT, BAUD, timeout=1)

print("Logging CSI to", OUT)
print("Recording time:", DURATION, "seconds")

header_written = False
packet_count = 0

start_time = time.time()

with open(OUT, "w", buffering=1) as f:

    while True:

        # stop after 3 minutes
        if time.time() - start_time >= DURATION:
            print("\nFinished collecting CSI packets")
            print("Total packets collected:", packet_count)

            if packet_count >= MIN_PACKETS:
                print("✅ Dataset is sufficient (>= 200 packets)")
            else:
                print("⚠️ Warning: Less than 200 packets collected")

            print("Saved to:", OUT)
            break

        try:
            line = ser.readline().decode(errors="ignore").strip()
        except:
            continue

        if not line:
            continue

        # detect header
        if line.startswith("type,role,mac") and not header_written:
            f.write(line + "\n")
            header_written = True
            print("Header written")
            continue

        # detect CSI row
        if "CSI_DATA" in line and "[" in line and "]":

            if not header_written:
                print("⚠️ waiting for header first...")
                continue

            f.write(line + "\n")

            packet_count += 1
            print("CSI packet:", packet_count)