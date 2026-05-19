import serial

PORT = "COM10"     # change if needed
BAUD = 115200
OUT  = "live_csi1.csv"

ser = serial.Serial(PORT, BAUD, timeout=1)

print("Logging CSI to", OUT)

header_written = False

with open(OUT, "w", buffering=1) as f:

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
        except:
            continue

        if not line:
            continue

        # detect header line from firmware
        if line.startswith("type,role,mac") and not header_written:
            f.write(line + "\n")
            header_written = True
            print("Header written")
            continue

        # detect CSI data row
        if "CSI_DATA" in line and "[" in line and "]" in line:
            if not header_written:
                print("⚠️ waiting for header first...")
                continue

            f.write(line + "\n")
            print("CSI row logged")
